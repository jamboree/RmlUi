#pragma once

#include "RmlUi_Renderer_VX.h"

struct PhysicalDeviceInfo {
    vk::PhysicalDeviceProperties m_Properties;
    vx::List<vk::QueueFamilyProperties> m_QueueFamilyProperties;
    vx::List<vk::ExtensionProperties> m_ExtensionProperties;

    bool Init(vx::PhysicalDevice physicalDevice);

    bool HasExtension(std::string_view name) const noexcept;
};

struct SyncObject {
    vk::Semaphore m_AcquireSemaphore;
    vk::Semaphore m_RenderSemaphore;
    vk::Fence m_RenderFence;
};

struct FrameResource {
    vk::ImageView m_ImageView;
    vk::Framebuffer m_Framebuffer;
};

struct ImageAttachment {
    vk::Image m_Image;
    vk::ImageView m_ImageView;
    vma::Allocation m_Allocation;
};

struct GfxContext_VX : RenderContext_VX {
    static constexpr uint32_t VulkanApiVersion = VK_API_VERSION_1_3;
    static constexpr uint32_t InFlightCount = 2;
    static constexpr const char* const RequiredDeviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};

    struct DeviceFeatures;

    vx::Instance m_Instance;
#ifndef NDEBUG
    vk::DebugUtilsMessengerEXT m_DebugMessenger;
#endif
    vk::SurfaceKHR m_Surface;
    vx::PhysicalDevice m_PhysicalDevice;
    vx::Device m_Device;
    vk::Queue m_Queue;
    vma::Allocator m_Allocator;
    vk::CommandPool m_CommandPool;
    vk::CommandPool m_TempCommandPool;
    vk::CommandBuffer m_CommandBuffers[InFlightCount];
    vk::DescriptorPool m_DescriptorPool;
    vk::RenderPass m_RenderPass;
    SyncObject m_SyncObjects[InFlightCount];
    vk::Semaphore m_TempSemaphore;
    vk::SwapchainKHR m_Swapchain;
    vx::List<FrameResource> m_FrameResources;
    ImageAttachment m_DepthStencilImage;

    vk::Format m_SwapchainImageFormat = vk::Format::eB8G8R8A8Unorm;
    vk::Format m_DepthStencilImageFormat = vk::Format::eD24UnormS8Uint;
    vk::Extent2D m_FrameExtent;
    uint32_t m_FrameNumber = 0;
    uint32_t m_ImageIndex = 0;
    uint32_t m_QueueFamilyIndex = 0;
    bool m_RenderTargetOutdated = false;

    Renderer_VX m_Renderer;

    void DestroyFrameResources();

    void DestroyImageAttachment(const ImageAttachment& res) {
        if (res.m_ImageView) {
            m_Device.destroyImageView(res.m_ImageView);
        }
        if (res.m_Image) {
            m_Allocator.destroyImage(res.m_Image, res.m_Allocation);
        }
    }

    void Destroy();

    void BeginFrame(vk::Extent2D extent);

    void EndFrame();

    void RecreateRenderTarget(vk::Extent2D extent);

    void InitInstance(std::vector<const char*>& extensions);

    bool InitContext();

    void InitRenderTarget(vk::Extent2D extent);

    bool QueryPhysicalDevice(vx::PhysicalDevice physicalDevice,
                             PhysicalDeviceInfo& deviceInfo) const;

    vk::PhysicalDevice SelectPhysicalDevice(PhysicalDeviceInfo& deviceInfo,
                                            DeviceFeatures& features);

    void InitDevice(PhysicalDeviceInfo& physicalDeviceInfo,
                    vk::PhysicalDeviceFeatures2& features);

    void InitRenderPass();

    void InitSyncObjects();

    void BuildSwapchain(const vk::SurfaceCapabilitiesKHR& capabilities);

    void BuildDepthStencilImage();

    void BuildFrameResources();

    void UpdateExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                      vk::Extent2D extent);

    uint32_t FindQueueFamilyIndex(
        const vx::List<vk::QueueFamilyProperties>& queueFamilyProperties,
        vk::QueueFlags flags, vk::SurfaceKHR surface = {}) const;

    vx::Device GetDevice() override { return m_Device; }

    vma::Allocator GetAllocator() override { return m_Allocator; }

    vk::Extent2D GetFrameExtent() override { return m_FrameExtent; }

    vx::CommandBuffer BeginTemp() override;

    void EndTemp(vk::CommandBuffer commandBuffer) override;
};
