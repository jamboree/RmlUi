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

struct ImageResource {
    vk::Image m_Image;
    vk::ImageView m_ImageView;
    vma::Allocation m_Allocation;
};

struct GfxContext_VX {
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
    vk::CommandBuffer m_CommandBuffers[InFlightCount + 1];
    vk::DescriptorPool m_DescriptorPool;
    vk::RenderPass m_RenderPass;
    SyncObject m_SyncObjects[InFlightCount];
    vk::Fence m_ImmediateFence;
    vk::SwapchainKHR m_Swapchain;
    vx::List<FrameResource> m_FrameResources;
    ImageResource m_DepthStencilImage;

    vk::Format m_SwapchainImageFormat = vk::Format::eB8G8R8A8Unorm;
    vk::Format m_DepthStencilImageFormat = vk::Format::eD24UnormS8Uint;
    vk::Extent2D m_FrameExtent;
    uint32_t m_FrameNumber = 0;
    uint32_t m_ImageIndex = 0;
    uint32_t m_QueueFamilyIndex = 0;
    bool m_RenderTargetOutdated = false;

    Renderer_VX m_Renderer;

    void DestroyFrameResources();

    void DestroyDepthStencilImage();

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

    void BuildSwapchain(vk::Extent2D extent);

    void BuildDepthStencilImage();

    void BuildFrameResources();

    void UpdateExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                      vk::Extent2D extent);

    uint32_t FindQueueFamilyIndex(
        const vx::List<vk::QueueFamilyProperties>& queueFamilyProperties,
        vk::QueueFlags flags, vk::SurfaceKHR surface = {}) const;

    vx::CommandBuffer BeginTransfer();

    void EndTransfer();

    static GfxContext_VX* GetContextPtr(void* p) {
        return reinterpret_cast<GfxContext_VX*>(
            static_cast<uint8_t*>(p) - offsetof(GfxContext_VX, m_Renderer));
    }

    static vx::Device GetDeviceImpl(Renderer_VX* p) {
        return GetContextPtr(p)->m_Device;
    }

    static vma::Allocator GetAllocatorImpl(Renderer_VX* p) {
        return GetContextPtr(p)->m_Allocator;
    }

    static vk::Extent2D GetFrameExtentImpl(Renderer_VX* p) {
        return GetContextPtr(p)->m_FrameExtent;
    }

    static vx::CommandBuffer BeginTransferImpl(Renderer_VX* p) {
        return GetContextPtr(p)->BeginTransfer();
    }

    static void EndTransferImpl(Renderer_VX* p) {
        GetContextPtr(p)->EndTransfer();
    }

    static constexpr Renderer_VX::Backend g_BackendImpl{
        .GetDevice = GetDeviceImpl,
        .GetAllocator = GetAllocatorImpl,
        .GetFrameExtent = GetFrameExtentImpl,
        .BeginTransfer = BeginTransferImpl,
        .EndTransfer = EndTransferImpl};
};
