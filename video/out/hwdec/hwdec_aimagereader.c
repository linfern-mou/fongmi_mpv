/*
 * Copyright (c) 2021 sfan5 <sfan5@live.de>
 *
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

#include "config.h"

#include <dlfcn.h>
#include <errno.h>
#include <poll.h>
#include <unistd.h>
#include <EGL/egl.h>
#include <android/data_space.h>
#include <media/NdkImageReader.h>
#include <android/native_window_jni.h>
#include <libavcodec/mediacodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_mediacodec.h>

#include "misc/jni.h"
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "video/out/gpu/hwdec.h"
#include "video/out/opengl/ra_gl.h"

#if HAVE_VULKAN
#include "hwdec_aimagereader_vk.h"
#endif

typedef void *GLeglImageOES;
typedef void *EGLImageKHR;
#define EGL_NATIVE_BUFFER_ANDROID 0x3140

struct priv_owner {
    struct mp_hwdec_ctx hwctx;
    AImageReader *reader;
    jobject java_surface;
    jmethodID surface_release_id;
    void *lib_handle;

    // AImageReader callbacks are queued and can outlive individual mappers.
    mp_mutex image_lock;
    mp_cond image_cond;
    bool image_available;

#if HAVE_VULKAN
    mp_mutex vk_lock;
    struct aimagereader_vk *vk;
#endif

    media_status_t (*AImageReader_newWithUsage)(
        int32_t, int32_t, int32_t, uint64_t, int32_t, AImageReader **);
    media_status_t (*AImageReader_getWindow)(
        AImageReader *, ANativeWindow **);
    media_status_t (*AImageReader_setImageListener)(
        AImageReader *, AImageReader_ImageListener *);
    media_status_t (*AImageReader_setBufferRemovedListener)(
        AImageReader *, AImageReader_BufferRemovedListener *);
    media_status_t (*AImageReader_acquireLatestImage)(
        AImageReader *, AImage **);
    media_status_t (*AImageReader_acquireLatestImageAsync)(
        AImageReader *, AImage **, int *);
    void (*AImageReader_delete)(AImageReader *);
    media_status_t (*AImage_getHardwareBuffer)(
        const AImage *, AHardwareBuffer **);
    media_status_t (*AImage_getCropRect)(
        const AImage *, AImageCropRect *);
    media_status_t (*AImage_getDataSpace)(
        const AImage *, int32_t *);
    void (*AImage_delete)(AImage *);
    void (*AImage_deleteAsync)(AImage *, int);
    void (*AHardwareBuffer_describe)(
        const AHardwareBuffer *, AHardwareBuffer_Desc *);
    jobject (*ANativeWindow_toSurface)(JNIEnv *, ANativeWindow *);
};

enum mapper_backend {
    MAPPER_BACKEND_GL,
    MAPPER_BACKEND_VULKAN,
};

struct priv {
    struct mp_log *log;
    enum mapper_backend backend;

    GLuint gl_texture;
    AImage *image;
    EGLImageKHR egl_image;

    EGLImageKHR (EGLAPIENTRY *CreateImageKHR)(
        EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLint *);
    EGLBoolean (EGLAPIENTRY *DestroyImageKHR)(EGLDisplay, EGLImageKHR);
    EGLClientBuffer (EGLAPIENTRY *GetNativeClientBufferANDROID)(
        const struct AHardwareBuffer *);
    void (EGLAPIENTRY *EGLImageTargetTexture2DOES)(GLenum, GLeglImageOES);

#if HAVE_VULKAN
    struct aimagereader_vk *vk;
#endif
};

#define LIB_FUNCTION(name) {#name, offsetof(struct priv_owner, name)}
static const struct {
    const char *symbol;
    int offset;
} lib_functions[] = {
    LIB_FUNCTION(AImageReader_newWithUsage),
    LIB_FUNCTION(AImageReader_getWindow),
    LIB_FUNCTION(AImageReader_setImageListener),
    LIB_FUNCTION(AImageReader_setBufferRemovedListener),
    LIB_FUNCTION(AImageReader_acquireLatestImage),
    LIB_FUNCTION(AImageReader_acquireLatestImageAsync),
    LIB_FUNCTION(AImageReader_delete),
    LIB_FUNCTION(AImage_getHardwareBuffer),
    LIB_FUNCTION(AImage_getCropRect),
    LIB_FUNCTION(AImage_delete),
    LIB_FUNCTION(AImage_deleteAsync),
    LIB_FUNCTION(AHardwareBuffer_describe),
    LIB_FUNCTION(ANativeWindow_toSurface),
    { NULL, 0 },
};
#undef LIB_FUNCTION

static void *resolve_lib_symbol(struct priv_owner *p, const char *symbol)
{
    void *function = dlsym(p->lib_handle, symbol);
    return function ? function : dlsym(RTLD_DEFAULT, symbol);
}

static AVBufferRef *create_mediacodec_device_ref(jobject surface)
{
    AVBufferRef *device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_MEDIACODEC);
    if (!device_ref)
        return NULL;

    AVHWDeviceContext *ctx = (void *)device_ref->data;
    AVMediaCodecDeviceContext *hwctx = ctx->hwctx;
    hwctx->surface = surface;

    if (av_hwdevice_ctx_init(device_ref) < 0)
        av_buffer_unref(&device_ref);

    return device_ref;
}

static bool load_lib_functions(struct priv_owner *p, struct mp_log *log)
{
    p->lib_handle = dlopen("libmediandk.so", RTLD_NOW | RTLD_GLOBAL);
    if (!p->lib_handle)
        return false;
    for (int i = 0; lib_functions[i].symbol; i++) {
        const char *sym = lib_functions[i].symbol;
        void *fun = resolve_lib_symbol(p, sym);
        if (!fun) {
            mp_warn(log, "Could not resolve symbol %s\n", sym);
            return false;
        }

        *(void **) ((uint8_t*)p + lib_functions[i].offset) = fun;
    }

    *(void **)&p->AImage_getDataSpace =
        resolve_lib_symbol(p, "AImage_getDataSpace");

    return true;
}

#if HAVE_VULKAN
static void buffer_removed_callback(void *context, AImageReader *reader,
                                    AHardwareBuffer *buffer)
{
    struct priv_owner *p = context;

    mp_mutex_lock(&p->vk_lock);
    if (p->vk)
        aimagereader_vk_buffer_removed(p->vk, buffer);
    mp_mutex_unlock(&p->vk_lock);
}
#endif

static void image_callback(void *context, AImageReader *reader)
{
    struct priv_owner *p = context;

    mp_mutex_lock(&p->image_lock);
    p->image_available = true;
    mp_cond_signal(&p->image_cond);
    mp_mutex_unlock(&p->image_lock);
}

static int init(struct ra_hwdec *hw)
{
    struct priv_owner *p = hw->priv;
    bool use_gl = ra_is_gl(hw->ra_ctx->ra);
    bool use_vulkan = false;

    mp_mutex_init(&p->image_lock);
    mp_cond_init(&p->image_cond);

#if HAVE_VULKAN
    mp_mutex_init(&p->vk_lock);
    use_vulkan = aimagereader_vk_available(hw->ra_ctx, hw->log);
#endif

    if (!use_gl && !use_vulkan)
        return -1;

    if (use_gl) {
        if (!eglGetCurrentContext())
            return -1;
        const char *exts =
            eglQueryString(eglGetCurrentDisplay(), EGL_EXTENSIONS);
        if (!gl_check_extension(exts, "EGL_ANDROID_image_native_buffer"))
            return -1;
    }

    JNIEnv *env = MP_JNI_GET_ENV(hw);
    if (!env)
        return -1;

    if (!load_lib_functions(p, hw->log))
        return -1;

    if (use_gl) {
        static const char *es2_exts[] = {"GL_OES_EGL_image_external", 0};
        static const char *es3_exts[] = {
            "GL_OES_EGL_image_external_essl3", 0
        };
        GL *gl = ra_gl_get(hw->ra_ctx->ra);
        if (gl_check_extension(gl->extensions, es3_exts[0]))
            hw->glsl_extensions = es3_exts;
        else
            hw->glsl_extensions = es2_exts;
    }

    // dummy dimensions, AImageReader only transports hardware buffers
    media_status_t ret = p->AImageReader_newWithUsage(16, 16,
        AIMAGE_FORMAT_PRIVATE, AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
        5, &p->reader);
    if (ret != AMEDIA_OK) {
        MP_ERR(hw, "newWithUsage failed: %d\n", ret);
        return -1;
    }
    mp_assert(p->reader);

    AImageReader_ImageListener image_listener = {
        .context = p,
        .onImageAvailable = image_callback,
    };
    ret = p->AImageReader_setImageListener(p->reader, &image_listener);
    if (ret != AMEDIA_OK) {
        MP_ERR(hw, "setImageListener failed: %d\n", ret);
        return -1;
    }

#if HAVE_VULKAN
    if (use_vulkan) {
        AImageReader_BufferRemovedListener buffer_listener = {
            .context = p,
            .onBufferRemoved = buffer_removed_callback,
        };
        ret = p->AImageReader_setBufferRemovedListener(
            p->reader, &buffer_listener);
        if (ret != AMEDIA_OK) {
            MP_ERR(hw, "setBufferRemovedListener failed: %d\n", ret);
            return -1;
        }
    }
#endif

    ANativeWindow *window;
    ret = p->AImageReader_getWindow(p->reader, &window);
    if (ret != AMEDIA_OK) {
        MP_ERR(hw, "getWindow failed: %d\n", ret);
        return -1;
    }
    mp_assert(window);

    // The Java wrapper owns a local native Surface reference. Release it
    // explicitly instead of deferring native cleanup to Java GC.
    jobject java_surface = p->ANativeWindow_toSurface(env, window);
    if (MP_JNI_EXCEPTION_LOG(hw) < 0 || !java_surface) {
        MP_JNI_LOCAL_FREEP(&java_surface);
        MP_ERR(hw, "Failed to create Java surface\n");
        return -1;
    }

    jclass surface_class = (*env)->GetObjectClass(env, java_surface);
    if (MP_JNI_EXCEPTION_LOG(hw) < 0 || !surface_class) {
        MP_JNI_LOCAL_FREEP(&surface_class);
        MP_JNI_LOCAL_FREEP(&java_surface);
        MP_ERR(hw, "Failed to resolve Java Surface class\n");
        return -1;
    }

    jmethodID surface_release_id =
        (*env)->GetMethodID(env, surface_class, "release", "()V");
    MP_JNI_LOCAL_FREEP(&surface_class);
    if (MP_JNI_EXCEPTION_LOG(hw) < 0 || !surface_release_id) {
        MP_JNI_LOCAL_FREEP(&java_surface);
        MP_ERR(hw, "Failed to resolve Java Surface.release\n");
        return -1;
    }

    p->java_surface = (*env)->NewGlobalRef(env, java_surface);
    MP_JNI_LOCAL_FREEP(&java_surface);
    if (MP_JNI_EXCEPTION_LOG(hw) < 0 || !p->java_surface) {
        MP_ERR(hw, "Failed to retain Java surface\n");
        return -1;
    }
    p->surface_release_id = surface_release_id;

    p->hwctx = (struct mp_hwdec_ctx) {
        .driver_name = hw->driver->name,
        .av_device_ref = create_mediacodec_device_ref(p->java_surface),
        .hw_imgfmt = IMGFMT_MEDIACODEC,
    };

    if (!p->hwctx.av_device_ref) {
        MP_VERBOSE(hw, "Failed to create hwdevice_ctx\n");
        return -1;
    }

    hwdec_devices_add(hw->devs, &p->hwctx);

    return 0;
}

static void uninit(struct ra_hwdec *hw)
{
    struct priv_owner *p = hw->priv;

    hwdec_devices_remove(hw->devs, &p->hwctx);
    av_buffer_unref(&p->hwctx.av_device_ref);

    if (p->java_surface) {
        JNIEnv *env = MP_JNI_GET_ENV(hw);
        mp_assert(env);
        mp_assert(p->surface_release_id);
        MP_JNI_CALL_VOID(p->java_surface, p->surface_release_id);
        MP_JNI_EXCEPTION_LOG(hw);
        MP_JNI_GLOBAL_FREEP(&p->java_surface);
        p->surface_release_id = NULL;
    }

    if (p->reader) {
        p->AImageReader_delete(p->reader);
        p->reader = NULL;
    }

    mp_mutex_destroy(&p->image_lock);
    mp_cond_destroy(&p->image_cond);

#if HAVE_VULKAN
    mp_assert(!p->vk);
    mp_mutex_destroy(&p->vk_lock);
#endif

    if (p->lib_handle) {
        dlclose(p->lib_handle);
        p->lib_handle = NULL;
    }
}

static int mapper_init(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;

    p->log = mapper->log;

    mapper->dst_params = mapper->src_params;
    mapper->dst_params.imgfmt = IMGFMT_RGB0;
    mapper->dst_params.hw_subfmt = 0;

#if HAVE_VULKAN
    struct priv_owner *o = mapper->owner->priv;
    if (!ra_is_gl(mapper->ra)) {
        const struct aimagereader_vk_api api = {
            .AImage_delete = o->AImage_delete,
            .AHardwareBuffer_describe = o->AHardwareBuffer_describe,
        };
        p->backend = MAPPER_BACKEND_VULKAN;
        p->vk = aimagereader_vk_create(mapper, &api);
        if (!p->vk)
            return -1;
        mp_mutex_lock(&o->vk_lock);
        mp_assert(!o->vk);
        o->vk = p->vk;
        mp_mutex_unlock(&o->vk_lock);
        return 0;
    }
#endif

    p->backend = MAPPER_BACKEND_GL;
    GL *gl = ra_gl_get(mapper->ra);
    p->CreateImageKHR = (void *)eglGetProcAddress("eglCreateImageKHR");
    p->DestroyImageKHR = (void *)eglGetProcAddress("eglDestroyImageKHR");
    p->GetNativeClientBufferANDROID =
        (void *)eglGetProcAddress("eglGetNativeClientBufferANDROID");
    p->EGLImageTargetTexture2DOES =
        (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    if (!p->CreateImageKHR || !p->DestroyImageKHR ||
        !p->GetNativeClientBufferANDROID || !p->EGLImageTargetTexture2DOES)
        return -1;

    // texture creation
    gl->GenTextures(1, &p->gl_texture);
    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, p->gl_texture);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, 0);

    struct ra_tex_params params = {
        .dimensions = 2,
        .w = mapper->src_params.w,
        .h = mapper->src_params.h,
        .d = 1,
        .format = ra_find_unorm_format(mapper->ra, 1, 4),
        .render_src = true,
        .src_linear = true,
        .external_oes = true,
    };

    if (params.format->ctype != RA_CTYPE_UNORM)
        return -1;

    mapper->tex[0] = ra_create_wrapped_tex(mapper->ra, &params, p->gl_texture);
    if (!mapper->tex[0])
        return -1;

    return 0;
}

static void mapper_uninit(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;

#if HAVE_VULKAN
    struct priv_owner *o = mapper->owner->priv;
    if (p->backend == MAPPER_BACKEND_VULKAN) {
        mp_mutex_lock(&o->vk_lock);
        mp_assert(o->vk == p->vk);
        o->vk = NULL;
        mp_mutex_unlock(&o->vk_lock);
        aimagereader_vk_destroy(&p->vk);
    } else
#endif
    {
        GL *gl = ra_gl_get(mapper->ra);
        if (p->gl_texture)
            gl->DeleteTextures(1, &p->gl_texture);
        p->gl_texture = 0;

        ra_tex_free(mapper->ra, &mapper->tex[0]);
    }
}

static void mapper_unmap(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

#if HAVE_VULKAN
    if (p->backend == MAPPER_BACKEND_VULKAN)
        return;
#endif

    if (p->egl_image) {
        p->DestroyImageKHR(eglGetCurrentDisplay(), p->egl_image);
        p->egl_image = 0;
    }

    if (p->image) {
        o->AImage_delete(p->image);
        p->image = NULL;
    }
}

static media_status_t acquire_latest_image(struct ra_hwdec_mapper *mapper,
                                           AImage **image,
                                           int *acquire_fence)
{
#if HAVE_VULKAN
    struct priv *p = mapper->priv;
#endif
    struct priv_owner *o = mapper->owner->priv;
    int64_t deadline = mp_time_ns() + MP_TIME_MS_TO_NS(100);

    while (true) {
        *image = NULL;
        *acquire_fence = -1;

        media_status_t ret;
#if HAVE_VULKAN
        if (p->backend == MAPPER_BACKEND_VULKAN) {
            ret = o->AImageReader_acquireLatestImageAsync(
                o->reader, image, acquire_fence);
        } else
#endif
        {
            ret = o->AImageReader_acquireLatestImage(o->reader, image);
        }
        if (ret != AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE)
            return ret;

        int64_t remaining = deadline - mp_time_ns();
        if (remaining <= 0)
            return ret;

        // acquireLatestImage() is allowed to consume all images covered by
        // already queued callbacks. Treat the listener as a wakeup hint and
        // retry until the shared deadline rather than failing on a stale one.
        mp_mutex_lock(&o->image_lock);
        if (!o->image_available)
            mp_cond_timedwait(&o->image_cond, &o->image_lock, remaining);
        o->image_available = false;
        mp_mutex_unlock(&o->image_lock);
    }
}

static int mapper_map(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    if (mapper->src->imgfmt != IMGFMT_MEDIACODEC)
        return -1;
#if HAVE_VULKAN
    if (p->backend == MAPPER_BACKEND_VULKAN &&
        aimagereader_vk_reuse(p->vk, mapper->src))
        return 0;
#endif
    AVMediaCodecBuffer *buffer = (AVMediaCodecBuffer *)mapper->src->planes[3];
    if (av_mediacodec_release_buffer(buffer, 1) < 0) {
        MP_ERR(mapper, "Failed rendering MediaCodec output buffer\n");
        return -1;
    }

    AImage *image = NULL;
    int acquire_fence = -1;
    media_status_t ret =
        acquire_latest_image(mapper, &image, &acquire_fence);
    if (ret == AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE) {
        MP_WARN(mapper, "Waiting for frame timed out!\n");
        // Keep the previous texture rather than flashing a render error.
        return mapper->tex[0] ? 0 : -1;
    }
    if (ret != AMEDIA_OK) {
        MP_ERR(mapper, "acquireLatestImage failed: %d\n", ret);
        return -1;
    }
    mp_assert(image);

    if (acquire_fence >= 0) {
        struct pollfd fence = {
            .fd = acquire_fence,
            .events = POLLIN,
        };
        int poll_result;
        do {
            poll_result = poll(&fence, 1, 100);
        } while (poll_result < 0 && (errno == EINTR || errno == EAGAIN));
        if (poll_result <= 0 || fence.revents & (POLLERR | POLLNVAL)) {
            MP_ERR(mapper, "Waiting for AImage acquire fence failed: "
                           "%d (revents 0x%x)\n",
                   poll_result, (unsigned)fence.revents);
            o->AImage_deleteAsync(image, acquire_fence);
            return -1;
        }
        close(acquire_fence);
    }

    AHardwareBuffer *hwbuf = NULL;
    ret = o->AImage_getHardwareBuffer(image, &hwbuf);
    if (ret != AMEDIA_OK) {
        MP_ERR(mapper, "getHardwareBuffer failed: %d\n", ret);
        o->AImage_delete(image);
        return -1;
    }
    mp_assert(hwbuf);

#if HAVE_VULKAN
    if (p->backend == MAPPER_BACKEND_VULKAN) {
        AImageCropRect crop;
        ret = o->AImage_getCropRect(image, &crop);
        if (ret != AMEDIA_OK) {
            MP_ERR(mapper, "getCropRect failed: %d\n", ret);
            o->AImage_delete(image);
            return -1;
        }

        int32_t data_space = ADATASPACE_UNKNOWN;
        if (o->AImage_getDataSpace &&
            o->AImage_getDataSpace(image, &data_space) != AMEDIA_OK)
            data_space = ADATASPACE_UNKNOWN;

        int result = aimagereader_vk_map(
            p->vk, image, hwbuf, &crop, data_space, mapper->src);
        if (result < 0)
            o->AImage_delete(image);
        return result;
    }
#endif

    GL *gl = ra_gl_get(mapper->ra);
    p->image = image;

    // Update texture size since it may differ
    AHardwareBuffer_Desc d;
    o->AHardwareBuffer_describe(hwbuf, &d);
    if (mapper->tex[0]->params.w != (int)d.width ||
        mapper->tex[0]->params.h != (int)d.height) {
        MP_VERBOSE(p, "Texture dimensions changed to %dx%d\n", d.width, d.height);
        mapper->tex[0]->params.w = d.width;
        mapper->tex[0]->params.h = d.height;
    }

    EGLClientBuffer buf = p->GetNativeClientBufferANDROID(hwbuf);
    if (!buf)
        return -1;

    const int attribs[] = {EGL_NONE};
    p->egl_image = p->CreateImageKHR(eglGetCurrentDisplay(),
        EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, buf, attribs);
    if (!p->egl_image)
        return -1;

    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, p->gl_texture);
    p->EGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, p->egl_image);
    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, 0);

    return 0;
}


const struct ra_hwdec_driver ra_hwdec_aimagereader = {
    .name = "aimagereader",
    .priv_size = sizeof(struct priv_owner),
    .imgfmts = {IMGFMT_MEDIACODEC, 0},
    .device_type = AV_HWDEVICE_TYPE_MEDIACODEC,
    .init = init,
    .uninit = uninit,
    .mapper = &(const struct ra_hwdec_mapper_driver){
        .priv_size = sizeof(struct priv),
        .init = mapper_init,
        .uninit = mapper_uninit,
        .map = mapper_map,
        .unmap = mapper_unmap,
    },
};
