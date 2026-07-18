/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <android/data_space.h>
#include <dlfcn.h>

#include "video/out/android_common.h"
#include "egl_helpers.h"
#include "common/common.h"
#include "context.h"
#include "utils.h"

typedef int32_t (*set_buffers_dataspace_fn)(ANativeWindow *window,
                                           int32_t dataspace);

struct priv {
    struct GL gl;
    EGLDisplay egl_display;
    EGLContext egl_context;
    EGLSurface egl_surface;
    EGLConfig egl_config;
    EGLint egl_native_visual_id;
    EGLint egl_colorspace;
    EGLint rejected_egl_colorspace;
    set_buffers_dataspace_fn set_buffers_dataspace;
    int32_t dataspace;
    int32_t rejected_dataspace;
    bool supports_pq;
    bool supports_hlg;
    bool supports_p3;
};

struct android_color {
    EGLint egl_colorspace;
    int32_t dataspace;
};

static bool android_get_color(struct priv *p, struct mp_image_params *params,
                              struct android_color *out)
{
    *out = (struct android_color) {
        .egl_colorspace = EGL_NONE,
        .dataspace = ADATASPACE_UNKNOWN,
    };
    if (!params)
        return true;

    struct pl_color_space *color = &params->color;
    switch (color->transfer) {
    case PL_COLOR_TRC_PQ:
        if (!p->supports_pq)
            return false;
        color->primaries = PL_COLOR_PRIM_BT_2020;
        out->egl_colorspace = EGL_GL_COLORSPACE_BT2020_PQ_EXT;
        out->dataspace = ADATASPACE_BT2020_PQ;
        return true;
    case PL_COLOR_TRC_HLG:
        if (!p->supports_hlg)
            return false;
        color->primaries = PL_COLOR_PRIM_BT_2020;
        out->egl_colorspace = EGL_GL_COLORSPACE_BT2020_HLG_EXT;
        out->dataspace = ADATASPACE_BT2020_HLG;
        return true;
    default:
        if (pl_color_transfer_is_hdr(color->transfer))
            return false;
        break;
    }

    switch (color->primaries) {
    case PL_COLOR_PRIM_DISPLAY_P3:
        if (!p->supports_p3)
            return false;
        color->transfer = PL_COLOR_TRC_SRGB;
        out->egl_colorspace = EGL_GL_COLORSPACE_DISPLAY_P3_EXT;
        out->dataspace = ADATASPACE_DISPLAY_P3;
        return true;
    case PL_COLOR_PRIM_BT_2020:
        color->transfer = PL_COLOR_TRC_BT_1886;
        out->dataspace = ADATASPACE_BT2020;
        return true;
    default:
        color->primaries = PL_COLOR_PRIM_BT_709;
        color->transfer = PL_COLOR_TRC_SRGB;
        out->dataspace = ADATASPACE_SRGB;
        return true;
    }
}

static bool android_set_egl_colorspace(struct ra_ctx *ctx,
                                       EGLint egl_colorspace);
static bool android_reconfig(struct ra_ctx *ctx);

static bool android_set_color(struct ra_ctx *ctx,
                              struct mp_image_params *params)
{
    struct priv *p = ctx->priv;
    ANativeWindow *native_window = vo_android_native_window(ctx->vo);
    if (!native_window)
        return false;

    struct android_color color;
    if (!android_get_color(p, params, &color))
        return false;
    if (!android_set_egl_colorspace(ctx, color.egl_colorspace))
        return false;

    if (color.egl_colorspace == EGL_NONE && color.dataspace != p->dataspace) {
        if (color.dataspace == p->rejected_dataspace ||
            !p->set_buffers_dataspace)
            return color.dataspace == ADATASPACE_UNKNOWN;

        int32_t ret = p->set_buffers_dataspace(native_window, color.dataspace);
        if (ret) {
            MP_VERBOSE(ctx, "Native window rejected data space %d: %d\n",
                       color.dataspace, ret);
            p->rejected_dataspace = color.dataspace;
            return false;
        }
    }

    p->dataspace = color.dataspace;
    p->rejected_dataspace = -1;
    return true;
}

static void android_reset_color_state(struct priv *p)
{
    p->egl_colorspace = EGL_NONE;
    p->rejected_egl_colorspace = -1;
    p->dataspace = ADATASPACE_UNKNOWN;
    p->rejected_dataspace = -1;
}

static void android_swap_buffers(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    if (!vo_android_has_native_window(ctx->vo) ||
        p->egl_surface == EGL_NO_SURFACE)
        return;

    eglSwapBuffers(p->egl_display, p->egl_surface);
}

static void android_destroy_egl_surface(struct ra_ctx *ctx, EGLSurface surface)
{
    struct priv *p = ctx->priv;
    if (!surface || surface == EGL_NO_SURFACE)
        return;

    eglDestroySurface(p->egl_display, surface);
}

static void android_destroy_current_egl_surface(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    if (!p->egl_surface || p->egl_surface == EGL_NO_SURFACE)
        return;

    eglMakeCurrent(p->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                   EGL_NO_CONTEXT);
    android_destroy_egl_surface(ctx, p->egl_surface);
    p->egl_surface = EGL_NO_SURFACE;
}

static EGLSurface android_create_egl_surface(struct ra_ctx *ctx,
                                             ANativeWindow *native_window,
                                             EGLint egl_colorspace)
{
    struct priv *p = ctx->priv;
    if (!native_window) {
        MP_ERR(ctx, "Missing native window\n");
        return EGL_NO_SURFACE;
    }

    ANativeWindow_setBuffersGeometry(native_window, 0, 0,
                                     p->egl_native_visual_id);

    const EGLint attrs[] = {
        EGL_GL_COLORSPACE_KHR, egl_colorspace,
        EGL_NONE,
    };
    EGLSurface egl_surface = eglCreateWindowSurface(p->egl_display,
                                                    p->egl_config,
                                                    (EGLNativeWindowType)native_window,
                                                    egl_colorspace == EGL_NONE
                                                        ? NULL : attrs);
    if (egl_surface == EGL_NO_SURFACE) {
        MP_VERBOSE(ctx, "Could not create EGL surface with color space %d: %d\n",
                   egl_colorspace, eglGetError());
        return EGL_NO_SURFACE;
    }

    if (egl_colorspace != EGL_NONE) {
        EGLint actual = EGL_NONE;
        if (!eglQuerySurface(p->egl_display, egl_surface,
                             EGL_GL_COLORSPACE_KHR, &actual) ||
            actual != egl_colorspace)
        {
            MP_VERBOSE(ctx, "EGL returned color space %d instead of %d\n",
                       actual, egl_colorspace);
            android_destroy_egl_surface(ctx, egl_surface);
            return EGL_NO_SURFACE;
        }
    }

    return egl_surface;
}

static bool android_make_current(struct ra_ctx *ctx, EGLSurface surface)
{
    struct priv *p = ctx->priv;
    if (!eglMakeCurrent(p->egl_display, surface, surface, p->egl_context)) {
        MP_ERR(ctx, "Failed to set context!\n");
        return false;
    }

    return true;
}

static bool android_create_current_egl_surface(struct ra_ctx *ctx,
                                               ANativeWindow *native_window,
                                               EGLint egl_colorspace)
{
    struct priv *p = ctx->priv;
    EGLSurface surface =
        android_create_egl_surface(ctx, native_window, egl_colorspace);
    if (surface == EGL_NO_SURFACE)
        return false;
    if (!android_make_current(ctx, surface)) {
        android_destroy_egl_surface(ctx, surface);
        return false;
    }

    p->egl_surface = surface;
    if (ctx->swapchain && !android_reconfig(ctx)) {
        android_destroy_current_egl_surface(ctx);
        return false;
    }
    return true;
}

static bool android_set_egl_colorspace(struct ra_ctx *ctx,
                                       EGLint egl_colorspace)
{
    struct priv *p = ctx->priv;
    if (egl_colorspace == p->egl_colorspace)
        return true;
    if (egl_colorspace == p->rejected_egl_colorspace)
        return false;

    ANativeWindow *native_window = vo_android_native_window(ctx->vo);
    if (!native_window)
        return false;

    EGLint previous = p->egl_colorspace;
    android_destroy_current_egl_surface(ctx);
    if (!android_create_current_egl_surface(ctx, native_window,
                                            egl_colorspace))
    {
        p->rejected_egl_colorspace = egl_colorspace;
        if (!android_create_current_egl_surface(ctx, native_window, previous))
            MP_ERR(ctx, "Could not restore the previous EGL surface\n");
        return false;
    }

    p->egl_colorspace = egl_colorspace;
    p->rejected_egl_colorspace = -1;
    p->dataspace = ADATASPACE_UNKNOWN;
    p->rejected_dataspace = -1;
    return true;
}

static bool android_check_visible(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    return vo_android_has_native_window(ctx->vo) &&
           p->egl_surface != EGL_NO_SURFACE;
}

static void android_uninit(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    ra_gl_ctx_uninit(ctx);

    android_destroy_current_egl_surface(ctx);
    if (p->egl_context)
        eglDestroyContext(p->egl_display, p->egl_context);

    vo_android_uninit(ctx->vo);
}

static bool android_init(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv = talloc_zero(ctx, struct priv);
    p->set_buffers_dataspace = (set_buffers_dataspace_fn)
        dlsym(RTLD_DEFAULT, "ANativeWindow_setBuffersDataSpace");
    android_reset_color_state(p);

    if (!vo_android_init(ctx->vo))
        goto fail;

    p->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(p->egl_display, NULL, NULL)) {
        MP_FATAL(ctx, "EGL failed to initialize.\n");
        goto fail;
    }

    const char *extensions = eglQueryString(p->egl_display, EGL_EXTENSIONS);
    p->supports_pq =
        gl_check_extension(extensions, "EGL_EXT_gl_colorspace_bt2020_pq");
    p->supports_hlg =
        gl_check_extension(extensions, "EGL_EXT_gl_colorspace_bt2020_hlg");
    p->supports_p3 =
        gl_check_extension(extensions, "EGL_EXT_gl_colorspace_display_p3");

    EGLConfig config;
    if (!mpegl_create_context(ctx, p->egl_display, &p->egl_context, &config))
        goto fail;
    p->egl_config = config;

    eglGetConfigAttrib(p->egl_display, p->egl_config, EGL_NATIVE_VISUAL_ID,
                       &p->egl_native_visual_id);
    ANativeWindow *native_window = vo_android_native_window(ctx->vo);
    if (!android_create_current_egl_surface(ctx, native_window, EGL_NONE))
        goto fail;

    mpegl_load_functions(&p->gl, ctx->log);

    struct ra_ctx_params params = {
        .check_visible = android_check_visible,
        .set_color = android_set_color,
        .swap_buffers = android_swap_buffers,
    };

    if (!ra_gl_ctx_init(ctx, &p->gl, params))
        goto fail;

    ctx->supports_auto_colorspace_hint =
        p->supports_pq || p->supports_hlg || p->supports_p3 ||
        p->set_buffers_dataspace;
    return true;
fail:
    android_uninit(ctx);
    return false;
}

static bool android_reconfig(struct ra_ctx *ctx)
{
    if (!vo_android_has_native_window(ctx->vo))
        return true;

    int w, h;
    if (!vo_android_surface_size(ctx->vo, &w, &h))
        return false;

    // Update window geometry to prevent screen tearing
    ANativeWindow *native_window = vo_android_native_window(ctx->vo);
    if (native_window) {
        int32_t current_format = ANativeWindow_getFormat(native_window);
        ANativeWindow_setBuffersGeometry(native_window, w, h, current_format);
    }

    ctx->vo->dwidth = w;
    ctx->vo->dheight = h;
    ra_gl_ctx_resize(ctx->swapchain, w, h, 0);
    return true;
}

static bool android_detach_window(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    android_destroy_current_egl_surface(ctx);
    vo_android_set_native_window(ctx->vo, NULL);
    android_reset_color_state(p);
    return true;
}

static bool android_update_window(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    if (ctx->vo->opts->WinID == 0 || ctx->vo->opts->WinID == -1)
        return android_detach_window(ctx);

    ANativeWindow *native_window = vo_android_create_native_window(ctx->vo);
    if (!native_window)
        return false;

    EGLSurface old_surface = p->egl_surface;
    EGLSurface new_surface =
        android_create_egl_surface(ctx, native_window, EGL_NONE);
    if (new_surface == EGL_NO_SURFACE)
        goto fail;
    if (!android_make_current(ctx, new_surface)) {
        android_destroy_egl_surface(ctx, new_surface);
        goto fail;
    }

    p->egl_surface = new_surface;
    android_destroy_egl_surface(ctx, old_surface);
    vo_android_set_native_window(ctx->vo, native_window);
    android_reset_color_state(p);
    if (!android_reconfig(ctx))
        return false;

    return true;

fail:
    ANativeWindow_release(native_window);
    return false;
}

static int android_control(struct ra_ctx *ctx, int *events, int request, void *arg)
{
    switch (request) {
    case VOCTRL_UPDATE_WINDOW:
        if (!android_update_window(ctx))
            return VO_FALSE;
        *events |= VO_EVENT_RESIZE | VO_EVENT_EXPOSE;
        return VO_TRUE;
    }

    return VO_NOTIMPL;
}

const struct ra_ctx_fns ra_ctx_android = {
    .type           = "opengl",
    .name           = "android",
    .description    = "Android/EGL",
    .reconfig       = android_reconfig,
    .control        = android_control,
    .init           = android_init,
    .uninit         = android_uninit,
};
