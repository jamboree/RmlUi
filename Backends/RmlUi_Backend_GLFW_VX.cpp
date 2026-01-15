#include "RmlUi_Backend.h"
#include "RmlUi_GfxContext_VX.h"
#include "RmlUi_Renderer_VX.h"
#include "RmlUi_Platform_GLFW.h"
#include <RmlUi/Config/Config.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <GLFW/glfw3.h>
#include <optional>

// Determines the anti-aliasing quality when creating layers. Enables
// better-looking visuals, especially when transforms are applied.
#ifndef RMLUI_NUM_MSAA_SAMPLES
#define RMLUI_NUM_MSAA_SAMPLES 2
#endif

struct BackendContext {
    GLFWwindow* m_Window = nullptr;
    std::optional<SystemInterface_GLFW> m_System;
    GfxContext_VX m_Gfx;
    Renderer_VX m_Renderer;

    Rml::Context* m_Context = nullptr;
    KeyDownCallback m_KeyDownCallback = nullptr;

    vk::Extent2D m_FrameExtent;
    int m_GlfwActiveModifiers = 0;

    bool Initialize(const char* window_name, int width, int height,
                    bool allow_resize) {
        m_System.emplace();
        glfwSetErrorCallback([](int error, const char* description) {
            Rml::Log::Message(Rml::Log::LT_ERROR, "GLFW error (0x%x): %s",
                              error, description);
        });
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, allow_resize ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
        m_Window =
            glfwCreateWindow(width, height, window_name, nullptr, nullptr);
        if (!m_Window) {
            Rml::Log::Message(Rml::Log::LT_ERROR,
                              "GLFW failed to create window");
            return false;
        }

        std::vector<const char*> extensions;
        {
            uint32_t count = 0;
            const auto list = glfwGetRequiredInstanceExtensions(&count);
            extensions.assign(list, list + count);
        }
        m_Gfx.InitInstance(extensions);
        check(vk::Result(glfwCreateWindowSurface(m_Gfx.m_Instance.handle,
                                                 m_Window, nullptr,
                                                 &m_Gfx.m_Surface.handle)));

        if (!m_Gfx.InitContext())
            return false;

        UpdateFramebufferSize();
        m_Gfx.m_SampleCount = vk::SampleCountFlagBits(RMLUI_NUM_MSAA_SAMPLES);
        m_Gfx.InitRenderTarget(m_FrameExtent);

        if (!m_Renderer.Init(m_Gfx)) {
            Rml::Log::Message(Rml::Log::LT_ERROR,
                              "Failed to initialize Vulkan render interface");
            return false;
        }

        m_System->SetWindow(m_Window);

        // Receive num lock and caps lock modifiers for proper handling of
        // numpad inputs in text fields.
        glfwSetInputMode(m_Window, GLFW_LOCK_KEY_MODS, GLFW_TRUE);

        SetupCallbacks();

        return true;
    }

    void UpdateFramebufferSize() {
        int width, height;
        glfwGetFramebufferSize(m_Window, &width, &height);
        m_FrameExtent.setWidth(width);
        m_FrameExtent.setHeight(height);
    }

    void Shutdown() {
        if (m_Gfx.m_Device) {
            (void)m_Gfx.m_Device.waitIdle();
        }
        m_Renderer.Destroy();
        m_Gfx.Destroy();
        if (m_Window) {
            glfwDestroyWindow(m_Window);
        }
        m_System.reset();
    }

    bool ProcessEvents(Rml::Context* context, KeyDownCallback key_down_callback,
                       bool power_save) {
        m_Context = context;
        m_KeyDownCallback = key_down_callback;

        if (power_save) {
            glfwWaitEventsTimeout(
                Rml::Math::Min(context->GetNextUpdateDelay(), 10.0));
        } else {
            glfwPollEvents();
        }
        // In case the window is mimimized, the renderer cannot accept any
        // render calls. We keep the application inside this loop until we are
        // able to render.
        bool running = true;
        for (;;) {
            if (glfwWindowShouldClose(m_Window)) {
                running = false;
                break;
            }
            UpdateFramebufferSize();
            if (m_FrameExtent.width) {
                if (m_Gfx.m_RenderTargetOutdated) {
                    m_Gfx.RecreateRenderTarget(m_FrameExtent);
                    m_Renderer.ResetRenderTarget();
                }
                break;
            }
            glfwWaitEvents();
        }

        m_Context = nullptr;
        m_KeyDownCallback = nullptr;

        return running;
    }

    static Rml::Context* GetContext(GLFWwindow* window) {
        return static_cast<BackendContext*>(glfwGetWindowUserPointer(window))
            ->m_Context;
    }

    void SetupCallbacks() {
        glfwSetWindowUserPointer(m_Window, this);
        // Key input
        glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int glfw_key,
                                        int /*scancode*/, int glfw_action,
                                        int glfw_mods) {
            const auto self =
                static_cast<BackendContext*>(glfwGetWindowUserPointer(window));
            const auto context = self->m_Context;
            if (!context)
                return;

            // Store the active modifiers for later because GLFW doesn't provide
            // them in the callbacks to the mouse input events.
            self->m_GlfwActiveModifiers = glfw_mods;

            // Override the default key event callback to add global shortcuts
            // for the samples.
            KeyDownCallback key_down_callback = self->m_KeyDownCallback;

            switch (glfw_action) {
            case GLFW_PRESS:
            case GLFW_REPEAT: {
                const Rml::Input::KeyIdentifier key =
                    RmlGLFW::ConvertKey(glfw_key);
                const int key_modifier =
                    RmlGLFW::ConvertKeyModifiers(glfw_mods);
                float dp_ratio = 1.f;
                glfwGetWindowContentScale(self->m_Window, &dp_ratio, nullptr);

                // See if we have any global shortcuts that take priority over
                // the context.
                if (key_down_callback &&
                    !key_down_callback(context, key, key_modifier, dp_ratio,
                                       true))
                    break;
                // Otherwise, hand the event over to the context by calling the
                // input handler as normal.
                if (!RmlGLFW::ProcessKeyCallback(context, glfw_key, glfw_action,
                                                 glfw_mods))
                    break;
                // The key was not consumed by the context either, try keyboard
                // shortcuts of lower priority.
                if (key_down_callback &&
                    !key_down_callback(context, key, key_modifier, dp_ratio,
                                       false))
                    break;
            } break;
            case GLFW_RELEASE:
                RmlGLFW::ProcessKeyCallback(context, glfw_key, glfw_action,
                                            glfw_mods);
                break;
            }
        });

        glfwSetCharCallback(
            m_Window, [](GLFWwindow* window, unsigned int codepoint) {
                RmlGLFW::ProcessCharCallback(GetContext(window), codepoint);
            });

        glfwSetCursorEnterCallback(m_Window, [](GLFWwindow* window,
                                                int entered) {
            RmlGLFW::ProcessCursorEnterCallback(GetContext(window), entered);
        });

        // Mouse input
        glfwSetCursorPosCallback(m_Window, [](GLFWwindow* window, double xpos,
                                              double ypos) {
            const auto self =
                static_cast<BackendContext*>(glfwGetWindowUserPointer(window));
            RmlGLFW::ProcessCursorPosCallback(self->m_Context, window, xpos,
                                              ypos,
                                              self->m_GlfwActiveModifiers);
        });

        glfwSetMouseButtonCallback(m_Window, [](GLFWwindow* window, int button,
                                                int action, int mods) {
            const auto self =
                static_cast<BackendContext*>(glfwGetWindowUserPointer(window));
            self->m_GlfwActiveModifiers = mods;
            RmlGLFW::ProcessMouseButtonCallback(self->m_Context, button, action,
                                                mods);
        });

        glfwSetScrollCallback(m_Window, [](GLFWwindow* window,
                                           double /*xoffset*/, double yoffset) {
            const auto self =
                static_cast<BackendContext*>(glfwGetWindowUserPointer(window));
            RmlGLFW::ProcessScrollCallback(self->m_Context, yoffset,
                                           self->m_GlfwActiveModifiers);
        });

        // Window events
        glfwSetFramebufferSizeCallback(
            m_Window, [](GLFWwindow* window, int width, int height) {
                RmlGLFW::ProcessFramebufferSizeCallback(GetContext(window),
                                                        width, height);
            });

        glfwSetWindowContentScaleCallback(
            m_Window, [](GLFWwindow* window, float xscale, float /*yscale*/) {
                RmlGLFW::ProcessContentScaleCallback(GetContext(window),
                                                     xscale);
            });
    }

    void BeginFrame() {
        m_Gfx.WaitNextFrame();
        m_Renderer.ResetFrame(m_Gfx.m_FrameIndex);
        if (!m_Gfx.AcquireRenderTarget()) {
            m_Gfx.RecreateRenderTarget(m_FrameExtent);
            m_Renderer.ResetRenderTarget();
        }
        auto commandBuffer = m_Gfx.BeginFrame();
        m_Renderer.BeginFrame(commandBuffer);
    }

    void EndFrame() {
        m_Renderer.EndFrame();
        m_Gfx.EndFrame();
    }
};

static BackendContext g_BackendContext;

bool Backend::Initialize(const char* window_name, int width, int height,
                         bool allow_resize) {
    if (!glfwInit()) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to initialize GLFW");
        return false;
    }
    if (volkInitialize()) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to initialize Vulkan");
        return false;
    }
    return g_BackendContext.Initialize(window_name, width, height,
                                       allow_resize);
}

void Backend::Shutdown() {
    g_BackendContext.Shutdown();
    volkFinalize();
    glfwTerminate();
}

Rml::SystemInterface* Backend::GetSystemInterface() {
    return g_BackendContext.m_System.operator->();
}

Rml::RenderInterface* Backend::GetRenderInterface() {
    return &g_BackendContext.m_Renderer;
}

bool Backend::ProcessEvents(Rml::Context* context,
                            KeyDownCallback key_down_callback,
                            bool power_save) {
    return g_BackendContext.ProcessEvents(context, key_down_callback,
                                          power_save);
}

void Backend::RequestExit() {
    glfwSetWindowShouldClose(g_BackendContext.m_Window, GLFW_TRUE);
}

void Backend::BeginFrame() { g_BackendContext.BeginFrame(); }

void Backend::PresentFrame() { g_BackendContext.EndFrame(); }