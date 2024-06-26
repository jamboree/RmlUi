#include "RmlUi_Backend.h"
#include "RmlUi_Renderer_VX.h"
#include "RmlUi_Platform_GLFW.h"
#include <RmlUi/Config/Config.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <GLFW/glfw3.h>

struct DeviceFeatures : vk::PhysicalDeviceFeatures2 {
    vk::PhysicalDeviceSynchronization2Features m_Synchronization2;
    vk::PhysicalDeviceBufferDeviceAddressFeatures m_BufferDeviceAddress;
    vk::PhysicalDeviceDynamicRenderingFeatures m_DynamicRendering;

    DeviceFeatures() noexcept {
        chain(m_Synchronization2)
            .chain(m_BufferDeviceAddress)
            .chain(m_DynamicRendering);
    }

    bool Init(const DeviceFeatures& supported) {
        if (!supported.m_Synchronization2.getSynchronization2())
            return false;
        m_Synchronization2.setSynchronization2(true);
        if (!supported.m_BufferDeviceAddress.getBufferDeviceAddress())
            return false;
        m_BufferDeviceAddress.setBufferDeviceAddress(true);
        if (!supported.m_DynamicRendering.getDynamicRendering())
            return false;
        m_DynamicRendering.setDynamicRendering(true);
        return true;
    }
};

struct PhysicalDeviceInfo {
    vk::PhysicalDeviceProperties m_Properties;
    vx::List<vk::QueueFamilyProperties> m_QueueFamilyProperties;
    vx::List<vk::ExtensionProperties> m_ExtensionProperties;
    DeviceFeatures m_Features;

    bool Init(vx::PhysicalDevice physicalDevice) {
        physicalDevice.getProperties(&m_Properties);
        m_QueueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        if (!physicalDevice.enumerateDeviceExtensionProperties().extract(
                m_ExtensionProperties)) {
            return false;
        }
        std::ranges::sort(m_ExtensionProperties, std::ranges::less{},
                          [](const vk::ExtensionProperties& props) {
                              return props.getExtensionName();
                          });
        return true;
    }

    bool HasExtension(std::string_view name) const noexcept {
        const auto it = std::ranges::lower_bound(
            m_ExtensionProperties, name, std::ranges::less{},
            [](const vk::ExtensionProperties& props) {
                return props.getExtensionName();
            });
        return it != m_ExtensionProperties.end() &&
               it->getExtensionName() == name;
    }
};

struct SyncObject {
    vk::Semaphore m_AcquireSemaphore;
    vk::Semaphore m_RenderSemaphore;
    vk::Fence m_RenderFence;
};

struct SwapchainTarget {
    vk::ImageView m_ImageView;
    vk::Framebuffer m_Framebuffer;
};

struct ImageResource {
    vk::Image m_Image;
    vk::ImageView m_ImageView;
    vma::Allocation m_Allocation;
};

template<class T>
struct ManualLifetime {
    alignas(T) uint8_t m_Storage[sizeof(T)];

    void Create() { new (m_Storage) T; }

    T* Get() noexcept { return reinterpret_cast<T*>(m_Storage); }

    void Destroy() { Get()->~T(); }
};

struct BackendContext {
    static constexpr uint32_t InFlightCount = 2;

    const char* const m_RequiredDeviceExtensions[3] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};

    GLFWwindow* m_Window = nullptr;

    vx::Instance m_Instance;
    vk::DebugUtilsMessengerEXT m_DebugMessenger;
    vk::SurfaceKHR m_Surface;
    vx::PhysicalDevice m_PhysicalDevice;
    vx::Device m_Device;
    vk::Queue m_Queue;
    vma::Allocator m_Allocator;
    vk::CommandPool m_CommandPool;
    vk::CommandBuffer m_CommandBuffers[InFlightCount + 1];
    vk::RenderPass m_RenderPass;
    SyncObject m_SyncObjects[InFlightCount];
    vk::Fence m_ImmediateFence;
    vk::SwapchainKHR m_Swapchain;
    vx::List<SwapchainTarget> m_SwapchainTargets;
    ImageResource m_DepthStencilImage;

    ManualLifetime<SystemInterface_GLFW> m_System;
    Renderer_VX m_Renderer;

    Rml::Context* m_Context = nullptr;
    KeyDownCallback m_KeyDownCallback = nullptr;

    vk::Format m_SwapchainImageFormat = vk::Format::eB8G8R8A8Unorm;
    vk::Format m_DepthStencilImageFormat = vk::Format::eD24UnormS8Uint;
    vk::Extent2D m_FrameExtent;
    int m_GlfwActiveModifiers = 0;
    uint32_t m_FrameNumber = 0;
    uint32_t m_ImageIndex = 0;
    uint32_t m_QueueFamilyIndex = 0;
    bool m_RecreateSwapchain = false;
    bool m_SystemCreated = false;

    bool Initialize(const char* window_name, int width, int height,
                    bool allow_resize) {
        if (!glfwInit()) {
            return false;
        }
        m_System.Create();
        m_SystemCreated = true;
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
        if (volkInitialize()) {
            Rml::Log::Message(Rml::Log::LT_ERROR,
                              "Failed to initialize Vulkan");
            return false;
        }

        InitInstance();
        check(vk::Result(glfwCreateWindowSurface(m_Instance.handle, m_Window,
                                                 nullptr, &m_Surface.handle)));
        PhysicalDeviceInfo deviceInfo;
        m_PhysicalDevice = SelectPhysicalDevice(deviceInfo);
        if (!m_PhysicalDevice) {
            Rml::Log::Message(Rml::Log::LT_ERROR, "no capable device");
            return false;
        }
        InitDevice(deviceInfo);
        InitRenderPass();
        InitSyncObjects();
        BuildSwapchain();
        BuildDepthStencilImage();
        BuildSwapchainTargets();

        m_System.Get()->SetWindow(m_Window);

        if (!m_Renderer.Init(g_BackendImpl)) {
            Rml::Log::Message(Rml::Log::LT_ERROR,
                              "Failed to initialize Vulkan render interface");
            return false;
        }

        // Receive num lock and caps lock modifiers for proper handling of
        // numpad inputs in text fields.
        glfwSetInputMode(m_Window, GLFW_LOCK_KEY_MODS, GLFW_TRUE);

        SetupCallbacks();

        return true;
    }

    void DestroySwapchainTargets() {
        for (auto& swapchainTarget : m_SwapchainTargets) {
            if (swapchainTarget.m_Framebuffer) {
                m_Device.destroyFramebuffer(swapchainTarget.m_Framebuffer);
            }
            if (swapchainTarget.m_ImageView) {
                m_Device.destroyImageView(swapchainTarget.m_ImageView);
            }
        }
    }

    void DestroyDepthStencilImage() {
        if (m_DepthStencilImage.m_ImageView) {
            m_Device.destroyImageView(m_DepthStencilImage.m_ImageView);
        }
        if (m_DepthStencilImage.m_Image) {
            m_Allocator.destroyImage(m_DepthStencilImage.m_Image,
                                     m_DepthStencilImage.m_Allocation);
        }
    }

    void Shutdown() {
        if (m_Device) {
            (void)m_Device.waitIdle();
        }
        m_Renderer.Shutdown();
        DestroySwapchainTargets();
        DestroyDepthStencilImage();
        if (m_Swapchain) {
            m_Device.destroySwapchainKHR(m_Swapchain);
        }
        if (m_ImmediateFence) {
            m_Device.destroyFence(m_ImmediateFence);
        }
        for (auto& syncObject : m_SyncObjects) {
            if (syncObject.m_AcquireSemaphore) {
                m_Device.destroySemaphore(syncObject.m_AcquireSemaphore);
            }
            if (syncObject.m_RenderSemaphore) {
                m_Device.destroySemaphore(syncObject.m_RenderSemaphore);
            }
            if (syncObject.m_RenderFence) {
                m_Device.destroyFence(syncObject.m_RenderFence);
            }
        }
        if (m_RenderPass) {
            m_Device.destroyRenderPass(m_RenderPass);
        }
        if (m_CommandPool) {
            m_Device.destroyCommandPool(m_CommandPool);
        }
        if (m_Allocator) {
            m_Allocator.destroy();
        }
        if (m_Device) {
            m_Device.destroy();
        }
        if (m_Surface) {
            m_Instance.destroySurfaceKHR(m_Surface);
        }
        if (m_DebugMessenger) {
            m_Instance.destroyDebugUtilsMessengerEXT(m_DebugMessenger);
        }
        if (m_Instance) {
            m_Instance.destroy();
        }
        if (m_Window) {
            glfwDestroyWindow(m_Window);
        }
        if (m_SystemCreated) {
            m_System.Destroy();
        }
        glfwTerminate();
    }

    void BeginFrame() {
        const auto& syncObject = m_SyncObjects[m_FrameNumber];
        const vx::CommandBuffer commandBuffer{m_CommandBuffers[m_FrameNumber]};
        check(m_Device.waitForFences(1, &syncObject.m_RenderFence, true,
                                     UINT64_MAX));
        if (auto ret = m_Device.acquireNextImageKHR(
                m_Swapchain, UINT64_MAX, syncObject.m_AcquireSemaphore);
            ret.result == vk::Result::eErrorOutOfDateKHR) {
            RecreateSwapchain();
        } else {
            if (ret.result == vk::Result::eSuboptimalKHR) {
                ret.result = vk::Result::eSuccess;
                m_RecreateSwapchain = true;
            }
            m_ImageIndex = ret.get();
        }
        vk::CommandBufferBeginInfo beginInfo;
        beginInfo.setFlags(vk::CommandBufferUsageFlagBits::bOneTimeSubmit);
        check(commandBuffer.begin(beginInfo));
        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.setRenderPass(m_RenderPass);
        renderPassBeginInfo.setFramebuffer(
            m_SwapchainTargets[m_ImageIndex].m_Framebuffer);
        renderPassBeginInfo.setRenderArea({{0, 0}, m_FrameExtent});
        const vk::ClearValue clearValues[2] = {
            {.color = vk::ClearColorValue()},
            {.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}}};
        renderPassBeginInfo.setClearValueCount(
            uint32_t(std::size(clearValues)));
        renderPassBeginInfo.setClearValues(clearValues);
        commandBuffer.cmdBeginRenderPass(renderPassBeginInfo,
                                         vk::SubpassContents::eInline);
        vk::Viewport viewport;
        viewport.setWidth(float(m_FrameExtent.getWidth()));
        viewport.setHeight(float(m_FrameExtent.getHeight()));
        viewport.setMinDepth(0.f);
        viewport.setMaxDepth(1.f);
        commandBuffer.cmdSetViewport(0, 1, &viewport);
        m_Renderer.BeginFrame(commandBuffer);
    }

    void EndFrame() {
        const auto& syncObject = m_SyncObjects[m_FrameNumber];
        const auto commandBuffer = m_CommandBuffers[m_FrameNumber];

        m_Renderer.EndFrame();
        commandBuffer.cmdEndRenderPass();
        check(commandBuffer.end());

        vk::CommandBufferSubmitInfo bufferSubmitInfo;
        bufferSubmitInfo.setCommandBuffer(commandBuffer);
        vk::SemaphoreSubmitInfo acquireSemaphoreInfo;
        acquireSemaphoreInfo.setSemaphore(syncObject.m_AcquireSemaphore);
        acquireSemaphoreInfo.setStageMask(
            vk::PipelineStageFlagBits2::bColorAttachmentOutput);
        vk::SemaphoreSubmitInfo renderSemaphoreInfo;
        renderSemaphoreInfo.setSemaphore(syncObject.m_RenderSemaphore);
        renderSemaphoreInfo.setStageMask(
            vk::PipelineStageFlagBits2::bAllGraphics);
        vk::SubmitInfo2 submitInfo;
        submitInfo.setCommandBufferInfoCount(1);
        submitInfo.setCommandBufferInfos(&bufferSubmitInfo);
        submitInfo.setWaitSemaphoreInfoCount(1);
        submitInfo.setWaitSemaphoreInfos(&acquireSemaphoreInfo);
        submitInfo.setSignalSemaphoreInfoCount(1);
        submitInfo.setSignalSemaphoreInfos(&renderSemaphoreInfo);

        check(m_Device.resetFences(1, &syncObject.m_RenderFence));
        check(m_Queue.submit2(1, &submitInfo, syncObject.m_RenderFence));

        vk::PresentInfoKHR presentInfo;
        presentInfo.setSwapchainCount(1);
        presentInfo.setSwapchains(&m_Swapchain);
        presentInfo.setImageIndices(&m_ImageIndex);
        presentInfo.setWaitSemaphoreCount(1);
        presentInfo.setWaitSemaphores(&syncObject.m_RenderSemaphore);

        if (const auto ret = m_Queue.presentKHR(presentInfo);
            ret == vk::Result::eErrorOutOfDateKHR ||
            ret == vk::Result::eSuboptimalKHR) {
            m_RecreateSwapchain = true;
            return;
        } else [[likely]] {
            check(ret);
        }

        m_FrameNumber = (m_FrameNumber + 1) % InFlightCount;
    }

    void RecreateSwapchain() {
        m_FrameNumber = 0;
        (void)m_Device.waitIdle();
        DestroySwapchainTargets();
        DestroyDepthStencilImage();
        m_SwapchainTargets.count = 0;
        const auto oldSwapchain = m_Swapchain;
        BuildSwapchain();
        m_Device.destroySwapchainKHR(oldSwapchain);
        BuildDepthStencilImage();
        BuildSwapchainTargets();
    }

    bool ProcessEvents(Rml::Context* context, KeyDownCallback key_down_callback,
                       bool power_save) {
        if (glfwWindowShouldClose(m_Window)) {
            return false;
        }
        m_Context = context;
        m_KeyDownCallback = key_down_callback;
        if (power_save) {
            glfwWaitEventsTimeout(
                Rml::Math::Min(context->GetNextUpdateDelay(), 10.0));
        } else {
            glfwPollEvents();
        }
        if (m_RecreateSwapchain) {
            RecreateSwapchain();
            m_RecreateSwapchain = false;
        }
        return true;
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

    void InitInstance() {
        vk::ApplicationInfo appInfo{/*applicationVersion*/ 0,
                                    /*engineVersion*/ 0,
                                    /*apiVersion*/ VK_API_VERSION_1_3};

        vk::InstanceCreateInfo instInfo;
        instInfo.setApplicationInfo(&appInfo);

        const char* const layers[] = {"VK_LAYER_KHRONOS_validation"};
        instInfo.setEnabledLayerCount(uint32_t(std::size(layers)));
        instInfo.setEnabledLayerNames(layers);
        std::vector<const char*> extensions;
        {
            uint32_t count = 0;
            const auto list = glfwGetRequiredInstanceExtensions(&count);
            extensions.assign(list, list + count);
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        instInfo.setEnabledExtensionCount(uint32_t(extensions.size()));
        instInfo.setEnabledExtensionNames(extensions.data());

        using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using Message = vk::DebugUtilsMessageTypeFlagBitsEXT;
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo;
        debugInfo.setMessageSeverity(Severity::bError);
        debugInfo.setMessageType(Message::bGeneral | Message::bValidation |
                                 Message::bPerformance);
        debugInfo.setPfnUserCallback(
            [](VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
               VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
               const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
               void* /*pUserData*/) -> VkBool32 {
                Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan error (%s): %s",
                                  pCallbackData->pMessageIdName,
                                  pCallbackData->pMessage);
                return false;
            });

        m_Instance = vk::createInstance(instInfo.chain(debugInfo)).get();
        volkLoadInstance(m_Instance.handle);
        m_DebugMessenger =
            m_Instance.createDebugUtilsMessengerEXT(debugInfo).get();
    }

    bool DiscoverPhysicalDevice(vx::PhysicalDevice physicalDevice,
                                PhysicalDeviceInfo& deviceInfo) const {
        if (!deviceInfo.Init(physicalDevice)) {
            return false;
        }
        for (const auto extension : m_RequiredDeviceExtensions) {
            if (!deviceInfo.HasExtension(extension)) {
                return false;
            }
        }
        bool hasSurfaceSupport = false;
        bool hasGraphics = false;
        const auto queueFamilyCount = deviceInfo.m_QueueFamilyProperties.count;
        for (uint32_t i = 0; i != queueFamilyCount; ++i) {
            const auto val = physicalDevice.getSurfaceSupportKHR(i, m_Surface);
            if (val.result == vk::Result::eSuccess && val.value) {
                hasSurfaceSupport = true;
            }
            const auto& prop = deviceInfo.m_QueueFamilyProperties[i];
            const auto flags = prop.getQueueFlags();
            if (flags & vk::QueueFlagBits::bGraphics) {
                hasGraphics = true;
            }
        }
        if (!hasSurfaceSupport || !hasGraphics) {
            return false;
        }
        DeviceFeatures supportedFeatures;
        physicalDevice.getFeatures2(&supportedFeatures);
        return deviceInfo.m_Features.Init(supportedFeatures);
    }

    vk::PhysicalDevice SelectPhysicalDevice(PhysicalDeviceInfo& deviceInfo) {
        vk::PhysicalDevice firstDevice;
        PhysicalDeviceInfo firstDeviceInfo;
        for (const auto physicalDevice :
             m_Instance.enumeratePhysicalDevices().get()) {
            if (DiscoverPhysicalDevice(physicalDevice, deviceInfo)) {
                if (deviceInfo.m_Properties.getDeviceType() ==
                    vk::PhysicalDeviceType::eDiscreteGpu) {
                    return physicalDevice;
                }
                if (!firstDevice) {
                    firstDevice = physicalDevice;
                    firstDeviceInfo = std::move(deviceInfo);
                }
            }
        }
        if (firstDevice) {
            deviceInfo = std::move(firstDeviceInfo);
        }
        return firstDevice;
    }

    void InitDevice(PhysicalDeviceInfo& physicalDeviceInfo) {
        m_QueueFamilyIndex = FindQueueFamilyIndex(
            physicalDeviceInfo.m_QueueFamilyProperties,
            vk::QueueFlagBits::bGraphics | vk::QueueFlagBits::bTransfer,
            m_Surface);
        const float queuePriorities[1] = {1.0f};
        vk::DeviceQueueCreateInfo queueInfo;
        queueInfo.setQueueFamilyIndex(m_QueueFamilyIndex);
        queueInfo.setQueueCount(1);
        queueInfo.setQueuePriorities(queuePriorities);

        const std::span extensions(m_RequiredDeviceExtensions);
        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfoCount(1);
        deviceInfo.setQueueCreateInfos(&queueInfo);
        deviceInfo.setEnabledExtensionCount(uint32_t(extensions.size()));
        deviceInfo.setEnabledExtensionNames(extensions.data());

        m_Device = m_PhysicalDevice
                       .createDevice(
                           deviceInfo.chainHead(physicalDeviceInfo.m_Features))
                       .get();
        m_Queue = m_Device.getQueue(m_QueueFamilyIndex, 0);

        vma::AllocatorCreateInfo allocatorInfo;
        allocatorInfo.setInstance(m_Instance);
        allocatorInfo.setPhysicalDevice(m_PhysicalDevice);
        allocatorInfo.setDevice(m_Device);
        allocatorInfo.setFlags(
            vma::AllocatorCreateFlagBits::bBufferDeviceAddress |
            vma::AllocatorCreateFlagBits::bMemoryBudgetEXT);
        allocatorInfo.setVulkanApiVersion(VK_API_VERSION_1_3);

        m_Allocator = vma::createAllocator(allocatorInfo).get();

        vk::CommandPoolCreateInfo poolInfo{m_QueueFamilyIndex};
        poolInfo.setFlags(vk::CommandPoolCreateFlagBits::bResetCommandBuffer);

        m_CommandPool = m_Device.createCommandPool(poolInfo).get();

        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setCommandPool(m_CommandPool);
        allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
        allocInfo.setCommandBufferCount(uint32_t(std::size(m_CommandBuffers)));

        check(m_Device.allocateCommandBuffers(allocInfo, m_CommandBuffers));
    }

    void InitRenderPass() {
        vk::AttachmentDescription2 attachmentDescriptions[2];
        attachmentDescriptions[0].setFormat(m_SwapchainImageFormat);
        attachmentDescriptions[0].setSamples(vk::SampleCountFlagBits::b1);
        attachmentDescriptions[0].setLoadOp(vk::AttachmentLoadOp::eClear);
        attachmentDescriptions[0].setStoreOp(vk::AttachmentStoreOp::eStore);
        attachmentDescriptions[0].setStencilLoadOp(
            vk::AttachmentLoadOp::eDontCare);
        attachmentDescriptions[0].setStencilStoreOp(
            vk::AttachmentStoreOp::eDontCare);
        attachmentDescriptions[0].setInitialLayout(vk::ImageLayout::eUndefined);
        attachmentDescriptions[0].setFinalLayout(
            vk::ImageLayout::ePresentSrcKHR);
        attachmentDescriptions[1].setFormat(m_DepthStencilImageFormat);
        attachmentDescriptions[1].setSamples(vk::SampleCountFlagBits::b1);
        attachmentDescriptions[1].setLoadOp(vk::AttachmentLoadOp::eClear);
        attachmentDescriptions[1].setStoreOp(vk::AttachmentStoreOp::eDontCare);
        attachmentDescriptions[1].setStencilLoadOp(
            vk::AttachmentLoadOp::eClear);
        attachmentDescriptions[1].setStencilStoreOp(
            vk::AttachmentStoreOp::eDontCare);
        attachmentDescriptions[1].setInitialLayout(vk::ImageLayout::eUndefined);
        attachmentDescriptions[1].setFinalLayout(
            vk::ImageLayout::eAttachmentOptimal);

        vk::SubpassDescription2 subpass;
        subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
        subpass.setColorAttachmentCount(1);

        vk::AttachmentReference2 colorAttachmentCount;
        colorAttachmentCount.setAttachment(0);
        colorAttachmentCount.setLayout(
            vk::ImageLayout::eColorAttachmentOptimal);
        colorAttachmentCount.setAspectMask(vk::ImageAspectFlagBits::bColor);
        subpass.setColorAttachments(&colorAttachmentCount);

        vk::AttachmentReference2 stencilAttachmentCount;
        stencilAttachmentCount.setAttachment(1);
        stencilAttachmentCount.setLayout(
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
        stencilAttachmentCount.setAspectMask(vk::ImageAspectFlagBits::bStencil);
        subpass.setDepthStencilAttachment(&stencilAttachmentCount);

        vk::SubpassDependency2 subpassDependency;
        subpassDependency.setSrcSubpass(VK_SUBPASS_EXTERNAL);
        subpassDependency.setDstSubpass(0);

        vk::RenderPassCreateInfo2 renderPassInfo;
        renderPassInfo.setSubpassCount(1);
        renderPassInfo.setSubpasses(&subpass);
        renderPassInfo.setAttachmentCount(
            uint32_t(std::size(attachmentDescriptions)));
        renderPassInfo.setAttachments(attachmentDescriptions);
        renderPassInfo.setDependencyCount(1);
        renderPassInfo.setDependencies(&subpassDependency);

        m_RenderPass = m_Device.createRenderPass2(renderPassInfo).get();
    }

    void InitSyncObjects() {
        vk::SemaphoreCreateInfo semaphoreInfo;
        vk::FenceCreateInfo fenceInfo;
        fenceInfo.setFlags(vk::FenceCreateFlagBits::bSignaled);
        for (auto& syncObject : m_SyncObjects) {
            syncObject.m_AcquireSemaphore =
                m_Device.createSemaphore(semaphoreInfo).get();
            syncObject.m_RenderSemaphore =
                m_Device.createSemaphore(semaphoreInfo).get();
            syncObject.m_RenderFence = m_Device.createFence(fenceInfo).get();
        }
        m_ImmediateFence = m_Device.createFence(fenceInfo).get();
    }

    void BuildSwapchain() {
        vk::SurfaceCapabilitiesKHR surfaceCapabilities;
        check(m_PhysicalDevice.getSurfaceCapabilitiesKHR(m_Surface,
                                                         &surfaceCapabilities));
        UpdateExtent(surfaceCapabilities);

        vk::SwapchainCreateInfoKHR swapchainInfo;
        swapchainInfo.setSurface(m_Surface);
        swapchainInfo.setMinImageCount(
            std::clamp(3u, surfaceCapabilities.getMinImageCount(),
                       surfaceCapabilities.getMaxImageCount()));
        swapchainInfo.setImageFormat(m_SwapchainImageFormat);
        swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
        swapchainInfo.setImageExtent(m_FrameExtent);
        swapchainInfo.setImageArrayLayers(1);
        swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::bColorAttachment |
                                    vk::ImageUsageFlagBits::bTransferDst |
                                    vk::ImageUsageFlagBits::bStorage);
        swapchainInfo.setImageSharingMode(vk::SharingMode::eExclusive);
        swapchainInfo.setPreTransform(
            vk::SurfaceTransformFlagBitsKHR::bIdentity);
        swapchainInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::bOpaque);
        swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
        swapchainInfo.setClipped(true);
        swapchainInfo.setOldSwapchain(m_Swapchain);

        m_Swapchain = m_Device.createSwapchainKHR(swapchainInfo).get();
    }

    void BuildDepthStencilImage() {
        const auto depthImageInfo = vx::image2DCreateInfo(
            m_DepthStencilImageFormat, m_FrameExtent,
            vk::ImageUsageFlagBits::bDepthStencilAttachment);

        vma::AllocationCreateInfo allocationInfo;
        allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

        m_DepthStencilImage.m_Image =
            m_Allocator
                .createImage(depthImageInfo, allocationInfo,
                             &m_DepthStencilImage.m_Allocation)
                .get();
        const auto imageViewInfo = vx::imageView2DCreateInfo(
            m_DepthStencilImage.m_Image, m_DepthStencilImageFormat,
            vk::ImageAspectFlagBits::bDepth |
                vk::ImageAspectFlagBits::bStencil);
        m_DepthStencilImage.m_ImageView =
            m_Device.createImageView(imageViewInfo).get();
    }

    void BuildSwapchainTargets() {
        const auto swapchainImages =
            m_Device.getSwapchainImagesKHR(m_Swapchain).get();
        m_SwapchainTargets.count = swapchainImages.count;
        m_SwapchainTargets.prepare();

        vk::FramebufferCreateInfo framebufferInfo;
        framebufferInfo.setRenderPass(m_RenderPass);
        framebufferInfo.setWidth(m_FrameExtent.getWidth());
        framebufferInfo.setHeight(m_FrameExtent.getHeight());
        framebufferInfo.setLayers(1);

        vk::ImageView attachments[2];
        attachments[1] = m_DepthStencilImage.m_ImageView;
        framebufferInfo.setAttachmentCount(uint32_t(std::size(attachments)));
        framebufferInfo.setAttachments(attachments);

        for (unsigned i = 0; i != swapchainImages.count; ++i) {
            const auto imageViewInfo = vx::imageView2DCreateInfo(
                swapchainImages[i], m_SwapchainImageFormat,
                vk::ImageAspectFlagBits::bColor);
            auto& swapchainTarget = m_SwapchainTargets[i];
            swapchainTarget.m_ImageView =
                m_Device.createImageView(imageViewInfo).get();
            attachments[0] = swapchainTarget.m_ImageView;
            swapchainTarget.m_Framebuffer =
                m_Device.createFramebuffer(framebufferInfo).get();
        }
    }

    void UpdateExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width == ~0u) {
            int width, height;
            glfwGetFramebufferSize(m_Window, &width, &height);
            const auto minExtent = capabilities.minImageExtent;
            const auto maxExtent = capabilities.maxImageExtent;
            m_FrameExtent.width =
                std::clamp(uint32_t(width), minExtent.width, maxExtent.width);
            m_FrameExtent.height = std::clamp(
                uint32_t(height), minExtent.height, maxExtent.height);
        } else [[likely]] {
            m_FrameExtent = capabilities.getCurrentExtent();
        }
    }

    uint32_t FindQueueFamilyIndex(
        const vx::List<vk::QueueFamilyProperties>& queueFamilyProperties,
        vk::QueueFlags flags, vk::SurfaceKHR surface = {}) const {
        unsigned extraBitCountMin = ~0u;
        uint32_t index = ~0u;
        for (uint32_t i = 0; i != queueFamilyProperties.count; ++i) {
            if (surface) {
                const auto val =
                    m_PhysicalDevice.getSurfaceSupportKHR(i, surface);
                if (val.result != vk::Result::eSuccess || !val.value)
                    continue;
            }
            const auto& prop = queueFamilyProperties[i];
            const auto queueFlags = prop.getQueueFlags();
            if (queueFlags.contains(flags)) {
                const unsigned extraBitCount =
                    std::popcount((queueFlags ^ flags).toUnderlying());
                if (extraBitCount < extraBitCountMin) {
                    extraBitCountMin = extraBitCount;
                    index = i;
                }
            }
        }
        return index;
    }

    vx::CommandBuffer BeginImmediateCommands() {
        const auto commandBuffer = m_CommandBuffers[InFlightCount];
        vk::CommandBufferBeginInfo beginInfo;
        beginInfo.setFlags(vk::CommandBufferUsageFlagBits::bOneTimeSubmit);
        check(commandBuffer.begin(beginInfo));
        return commandBuffer;
    }

    void EndImmediateCommands() {
        const auto commandBuffer = m_CommandBuffers[InFlightCount];
        check(commandBuffer.end());

        vk::CommandBufferSubmitInfo bufferSubmitInfo;
        bufferSubmitInfo.setCommandBuffer(commandBuffer);

        vk::SubmitInfo2 submitInfo;
        submitInfo.setCommandBufferInfoCount(1);
        submitInfo.setCommandBufferInfos(&bufferSubmitInfo);

        check(m_Device.resetFences(1, &m_ImmediateFence));
        check(m_Queue.submit2(1, &submitInfo, m_ImmediateFence));
        check(m_Device.waitForFences(1, &m_ImmediateFence, true, UINT64_MAX));
    }

    static BackendContext* GetBackendPtr(void* p) {
        return reinterpret_cast<BackendContext*>(
            static_cast<uint8_t*>(p) - offsetof(BackendContext, m_Renderer));
    }

    static vx::Device GetDeviceImpl(Renderer_VX* p) {
        return GetBackendPtr(p)->m_Device;
    }

    static vma::Allocator GetAllocatorImpl(Renderer_VX* p) {
        return GetBackendPtr(p)->m_Allocator;
    }

    static vk::RenderPass GetRenderPassImpl(Renderer_VX* p) {
        return GetBackendPtr(p)->m_RenderPass;
    }

    static vk::Extent2D GetFrameExtentImpl(Renderer_VX* p) {
        return GetBackendPtr(p)->m_FrameExtent;
    }

    static vx::CommandBuffer BeginCommandsImpl(Renderer_VX* p) {
        return GetBackendPtr(p)->BeginImmediateCommands();
    }

    static void EndCommandsImpl(Renderer_VX* p) {
        GetBackendPtr(p)->EndImmediateCommands();
    }

    static constexpr Renderer_VX::Backend g_BackendImpl{
        .GetDevice = GetDeviceImpl,
        .GetAllocator = GetAllocatorImpl,
        .GetRenderPass = GetRenderPassImpl,
        .GetFrameExtent = GetFrameExtentImpl,
        .BeginCommands = BeginCommandsImpl,
        .EndCommands = EndCommandsImpl};
};

static BackendContext g_BackendContext;

bool Backend::Initialize(const char* window_name, int width, int height,
                         bool allow_resize) {
    return g_BackendContext.Initialize(window_name, width, height,
                                       allow_resize);
}

void Backend::Shutdown() { g_BackendContext.Shutdown(); }

Rml::SystemInterface* Backend::GetSystemInterface() {
    return g_BackendContext.m_System.Get();
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