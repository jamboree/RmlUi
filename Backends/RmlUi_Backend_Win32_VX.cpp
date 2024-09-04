#include "RmlUi_Backend.h"
#define UNICODE
#include "RmlUi_GfxContext_VX.h"
#include "RmlUi_Platform_Win32.h"
#include <RmlUi/Config/Config.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/Profiling.h>
#include <optional>

/**
        High DPI support using Windows Per Monitor V2 DPI awareness.

        Requires Windows 10, version 1703.
 */
static bool g_HasDpiSupport = false;

// Make ourselves DPI aware on supported Windows versions.
static void InitializeDpiSupport() {
    if (::SetProcessDpiAwarenessContext(
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
        g_HasDpiSupport = true;
}

static UINT GetWindowDpi(HWND window_handle) {
    if (g_HasDpiSupport) {
        UINT dpi = ::GetDpiForWindow(window_handle);
        if (dpi != 0)
            return dpi;
    }
    return USER_DEFAULT_SCREEN_DPI;
}

static float GetDensityIndependentPixelRatio(HWND window_handle) {
    return float(GetWindowDpi(window_handle)) / float(USER_DEFAULT_SCREEN_DPI);
}

static void WaitEventsTimeout(UINT timeout) {
    MSG message;
    UINT_PTR timer_id = ::SetTimer(nullptr, 0, timeout, nullptr);
    BOOL res = ::GetMessage(&message, nullptr, 0, 0);
    ::KillTimer(nullptr, timer_id);
    if (!res)
        return;
    if (message.message != WM_TIMER || message.hwnd != nullptr ||
        message.wParam != timer_id) {
        ::TranslateMessage(&message);
        ::DispatchMessage(&message);
    }
}

static void PollEvents() {
    MSG message;
    while (::PeekMessage(&message, nullptr, 0, 0, PM_REMOVE)) {
        ::TranslateMessage(&message);
        ::DispatchMessage(&message);
    }
}

static void WaitEvents() {
    MSG message;
    if (::GetMessage(&message, nullptr, 0, 0)) {
        ::TranslateMessage(&message);
        ::DispatchMessage(&message);
    }
}

struct BackendContext {
    std::optional<SystemInterface_Win32> m_System;
    GfxContext_VX m_Gfx;
    TextInputMethodEditor_Win32 m_TextIme;

    HINSTANCE m_InstanceHnd = nullptr;
    std::wstring m_InstanceName;
    HWND m_WindowHnd = nullptr;

    Rml::Vector2i m_WindowSize;
    bool context_dimensions_dirty = true;
    bool m_Running = true;

    // Arguments set during event processing and nulled otherwise.
    Rml::Context* m_Context = nullptr;
    KeyDownCallback m_KeyDownCallback = nullptr;

    vk::Extent2D GetFrameExtent() const {
        return vk::Extent2D(m_WindowSize.x, m_WindowSize.y);
    }

    bool Initialize(const char* window_name, int width, int height,
                    bool allow_resize) {
        m_System.emplace();
        m_InstanceHnd = GetModuleHandle(nullptr);
        m_InstanceName = RmlWin32::ConvertToUTF16(Rml::String(window_name));

        InitializeDpiSupport();

        // Initialize the window but don't show it yet.
        m_WindowHnd = InitializeWindow(width, height, allow_resize);
        if (!m_WindowHnd)
            return false;

        std::vector<const char*> extensions;
        {
            extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
            extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        }
        m_Gfx.InitInstance(extensions);

        vk::Win32SurfaceCreateInfoKHR win32SurfaceInfo;
        win32SurfaceInfo.setHinstance(m_InstanceHnd);
        win32SurfaceInfo.setHwnd(m_WindowHnd);

        if (const auto ret =
                m_Gfx.m_Instance.createWin32SurfaceKHR(win32SurfaceInfo);
            ret.result == vk::Result::eSuccess) {
            m_Gfx.m_Surface = ret.value;
        } else {
            Rml::Log::Message(Rml::Log::LT_ERROR,
                              "Failed to initialize Vulkan render interface");
            ::CloseWindow(m_WindowHnd);
            return false;
        }

        if (!m_Gfx.InitContext())
            return false;

        m_WindowSize.x = width;
        m_WindowSize.y = height;
        m_Gfx.InitRenderTarget(GetFrameExtent());

        m_System->SetWindow(m_WindowHnd);

        // Provide a backend-specific text input handler to manage the IME.
        Rml::SetTextInputHandler(&m_TextIme);

        // Now we are ready to show the window.
        ::ShowWindow(m_WindowHnd, SW_SHOW);
        ::SetForegroundWindow(m_WindowHnd);
        ::SetFocus(m_WindowHnd);

        return true;
    }

    // Create the window but don't show it yet. Returns the pixel size of the
    // window, which may be different than the passed size due to DPI settings.
    HWND InitializeWindow(int& inout_width, int& inout_height,
                          bool allow_resize) {
        // Fill out the window class struct.
        WNDCLASSW window_class = {};
        window_class.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        window_class.lpfnWndProc =
            &WindowProcedureHandler; // Attach our local event handler.
        window_class.cbClsExtra = 0;
        window_class.cbWndExtra = 0;
        window_class.hInstance = m_InstanceHnd;
        window_class.hIcon = LoadIcon(nullptr, IDI_WINLOGO);
        window_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
        window_class.hbrBackground = nullptr;
        window_class.lpszMenuName = nullptr;
        window_class.lpszClassName = m_InstanceName.data();

        if (!RegisterClassW(&window_class)) {
            Rml::Log::Message(Rml::Log::LT_ERROR,
                              "Failed to register window class");
            return nullptr;
        }

        HWND window_handle = ::CreateWindowExW(
            WS_EX_APPWINDOW | WS_EX_WINDOWEDGE,
            m_InstanceName.data(), // Window class name.
            m_InstanceName.data(),
            WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_OVERLAPPEDWINDOW, 0,
            0,    // Window position.
            0, 0, // Window size.
            nullptr, nullptr, m_InstanceHnd, nullptr);

        if (!window_handle) {
            Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create window");
            return nullptr;
        }
        ::SetWindowLongPtr(window_handle, GWLP_USERDATA,
                           reinterpret_cast<LONG_PTR>(this));

        UINT window_dpi = GetWindowDpi(window_handle);
        inout_width = (inout_width * (int)window_dpi) / USER_DEFAULT_SCREEN_DPI;
        inout_height =
            (inout_height * (int)window_dpi) / USER_DEFAULT_SCREEN_DPI;

        DWORD style =
            (allow_resize
                 ? WS_OVERLAPPEDWINDOW
                 : (WS_OVERLAPPEDWINDOW & ~WS_SIZEBOX & ~WS_MAXIMIZEBOX));
        DWORD extended_style = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;

        // Adjust the window size to take the edges into account.
        RECT window_rect;
        window_rect.top = 0;
        window_rect.left = 0;
        window_rect.right = inout_width;
        window_rect.bottom = inout_height;
        if (g_HasDpiSupport)
            ::AdjustWindowRectExForDpi(&window_rect, style, FALSE,
                                       extended_style, window_dpi);
        else
            ::AdjustWindowRectEx(&window_rect, style, FALSE, extended_style);

        ::SetWindowLong(window_handle, GWL_EXSTYLE, extended_style);
        ::SetWindowLong(window_handle, GWL_STYLE, style);

        // Resize the window and center it on the screen.
        Rml::Vector2i screen_size = {GetSystemMetrics(SM_CXSCREEN),
                                     GetSystemMetrics(SM_CYSCREEN)};
        Rml::Vector2i window_size = {int(window_rect.right - window_rect.left),
                                     int(window_rect.bottom - window_rect.top)};
        Rml::Vector2i window_pos =
            Rml::Math::Max((screen_size - window_size) / 2, Rml::Vector2i(0));

        ::SetWindowPos(window_handle, HWND_TOP, window_pos.x, window_pos.y,
                       window_size.x, window_size.y, SWP_NOACTIVATE);

        return window_handle;
    }

    // Local event handler for window and input events.
    static LRESULT CALLBACK WindowProcedureHandler(HWND window_handle,
                                                   UINT message, WPARAM w_param,
                                                   LPARAM l_param) {
        const auto self = reinterpret_cast<BackendContext*>(
            ::GetWindowLongPtr(window_handle, GWLP_USERDATA));
        if (self) {
            switch (message) {
            case WM_CLOSE: {
                self->m_Running = false;
                return 0;
            } break;
            case WM_SIZE: {
                const int width = LOWORD(l_param);
                const int height = HIWORD(l_param);
                self->m_WindowSize.x = width;
                self->m_WindowSize.y = height;
                if (self->m_Context) {
                    self->m_Context->SetDimensions(self->m_WindowSize);
                }
                return 0;
            } break;
            case WM_DPICHANGED: {
                RECT* new_pos = (RECT*)l_param;
                SetWindowPos(window_handle, nullptr, new_pos->left,
                             new_pos->top, new_pos->right - new_pos->left,
                             new_pos->bottom - new_pos->top,
                             SWP_NOZORDER | SWP_NOACTIVATE);
                if (self->m_Context && g_HasDpiSupport)
                    self->m_Context->SetDensityIndependentPixelRatio(
                        GetDensityIndependentPixelRatio(window_handle));
                return 0;
            } break;
            case WM_KEYDOWN: {
                // Override the default key event callback to add global
                // shortcuts for the samples.
                Rml::Context* context = self->m_Context;
                KeyDownCallback key_down_callback = self->m_KeyDownCallback;

                const Rml::Input::KeyIdentifier rml_key =
                    RmlWin32::ConvertKey((int)w_param);
                const int rml_modifier = RmlWin32::GetKeyModifierState();
                const float native_dp_ratio =
                    GetDensityIndependentPixelRatio(window_handle);

                // See if we have any global shortcuts that take priority over
                // the context.
                if (key_down_callback &&
                    !key_down_callback(context, rml_key, rml_modifier,
                                       native_dp_ratio, true))
                    return 0;
                // Otherwise, hand the event over to the context by calling the
                // input handler as normal.
                if (!RmlWin32::WindowProcedure(context, self->m_TextIme,
                                               window_handle, message, w_param,
                                               l_param))
                    return 0;
                // The key was not consumed by the context either, try keyboard
                // shortcuts of lower priority.
                if (key_down_callback &&
                    !key_down_callback(context, rml_key, rml_modifier,
                                       native_dp_ratio, false))
                    return 0;
                return 0;
            } break;
            default: {
                // Submit it to the platform handler for default input handling.
                if (!RmlWin32::WindowProcedure(self->m_Context, self->m_TextIme,
                                               window_handle, message, w_param,
                                               l_param))
                    return 0;
            } break;
            }
        }
        // All unhandled messages go to DefWindowProc.
        return ::DefWindowProc(window_handle, message, w_param, l_param);
    }

    void Shutdown() {
        if (m_Gfx.m_Device) {
            (void)m_Gfx.m_Device.waitIdle();
        }
        // As we forcefully override the global text input handler, we must
        // reset it before the data is destroyed to avoid any potential
        // use-after-free.
        if (Rml::GetTextInputHandler() == &m_TextIme)
            Rml::SetTextInputHandler(nullptr);

        m_Gfx.Destroy();
        ::DestroyWindow(m_WindowHnd);
        ::UnregisterClassW((LPCWSTR)m_InstanceName.data(), m_InstanceHnd);
        m_System.reset();
    }

    bool ProcessEvents(Rml::Context* context, KeyDownCallback key_down_callback,
                       bool power_save) {
        // The initial window size may have been affected by system DPI
        // settings,
        // apply the actual pixel size and dp-ratio to the context.
        if (context_dimensions_dirty) {
            context_dimensions_dirty = false;
            const float dp_ratio = GetDensityIndependentPixelRatio(m_WindowHnd);
            context->SetDimensions(m_WindowSize);
            context->SetDensityIndependentPixelRatio(dp_ratio);
        }

        m_Context = context;
        m_KeyDownCallback = key_down_callback;

        if (power_save) {
            WaitEventsTimeout(
                Rml::Math::Min(context->GetNextUpdateDelay(), 10.0) * 1000);
        } else {
            PollEvents();
        }

        while (m_Running) {
            if (m_WindowSize.x) {
                if (m_Gfx.m_RenderTargetOutdated) {
                    m_Gfx.RecreateRenderTarget(GetFrameExtent());
                    m_Gfx.m_RenderTargetOutdated = false;
                }
                break;
            }
            WaitEvents();
        }

        m_Context = nullptr;
        m_KeyDownCallback = nullptr;

        return m_Running;
    }
};

static BackendContext g_BackendContext;

bool Backend::Initialize(const char* window_name, int width, int height,
                         bool allow_resize) {
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
}

Rml::SystemInterface* Backend::GetSystemInterface() {
    return g_BackendContext.m_System.operator->();
}

Rml::RenderInterface* Backend::GetRenderInterface() {
    return &g_BackendContext.m_Gfx.m_Renderer;
}

bool Backend::ProcessEvents(Rml::Context* context,
                            KeyDownCallback key_down_callback,
                            bool power_save) {
    RMLUI_ASSERT(context);
    return g_BackendContext.ProcessEvents(context, key_down_callback,
                                          power_save);
}

void Backend::RequestExit() { g_BackendContext.m_Running = false; }

void Backend::BeginFrame() {
    g_BackendContext.m_Gfx.BeginFrame(g_BackendContext.GetFrameExtent());
}

void Backend::PresentFrame() {
    g_BackendContext.m_Gfx.EndFrame();

    // Optional, used to mark frames during performance profiling.
    RMLUI_FrameMark;
}