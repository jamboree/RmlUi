#pragma once

#include "RmlUi_Include_VulkanX.h"
#include <vector>

struct PhysicalDeviceInfo {
    vk::PhysicalDeviceProperties m_Properties;
    vx::List<vk::QueueFamilyProperties> m_QueueFamilyProperties;
    vx::List<vk::ExtensionProperties> m_ExtensionProperties;

    bool Init(vx::PhysicalDevice physicalDevice);

    bool HasExtension(std::string_view name) const noexcept;
};

struct ImagePair {
    vk::Image m_Image;
    vk::ImageView m_ImageView;
};

struct ImageAttachment : ImagePair {
    vma::Allocation m_Allocation;
};

struct GfxContext_VX {
    static constexpr uint32_t VulkanApiVersion = VK_API_VERSION_1_3;
    static constexpr uint32_t InFlightCount = 2;
    static constexpr const char* const RequiredDeviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME};

    struct DeviceFeatures;

    struct PresentResource {
        vk::Image m_Image;
        vk::ImageView m_ImageView;
        vk::Semaphore m_RenderSemaphore;
    };

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
    vx::CommandBuffer m_CommandBuffers[InFlightCount];
    vk::Semaphore m_AcquireSemaphores[InFlightCount];
    vk::Semaphore m_TempSemaphore;
    vk::Fence m_RenderFences[InFlightCount];
    vk::SwapchainKHR m_Swapchain;
    vx::List<PresentResource> m_PresentResources;
    ImageAttachment m_DepthStencilImage;

    vk::Format m_SwapchainImageFormat = vk::Format::eB8G8R8A8Unorm;
    vk::Format m_DepthStencilImageFormat = vk::Format::eD24UnormS8Uint;
    vk::SampleCountFlagBits m_SampleCount = vk::SampleCountFlagBits::b1;
    vk::Extent2D m_FrameExtent;
    uint32_t m_FrameIndex = 0;
    uint32_t m_PresentIndex = 0;
    uint32_t m_QueueFamilyIndex = 0;
    bool m_RenderTargetOutdated = false;

    void DestroyPresentResources();

    void Destroy();

    const PresentResource& CurrentPresentResource() const {
        return m_PresentResources[m_PresentIndex];
    }

    void AcquireNextFrame() {
        m_FrameIndex = (m_FrameIndex + 1) % InFlightCount;
        check(m_Device.waitForFences(1, m_RenderFences + m_FrameIndex, true,
                                     UINT64_MAX));
    }

    bool InitFrame();

    vx::CommandBuffer BeginFrame();

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

    void InitAllocator();

    void InitCommandBuffers();

    void InitSyncObjects();

    void BuildSwapchain(const vk::SurfaceCapabilitiesKHR& capabilities);

    void BuildPresentResources();

    void UpdateExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                      vk::Extent2D extent);

    uint32_t FindQueueFamilyIndex(
        const vx::List<vk::QueueFamilyProperties>& queueFamilyProperties,
        vk::QueueFlags flags, vk::SurfaceKHR surface = {}) const;

    vx::CommandBuffer BeginTemp();

    void EndTemp(vk::CommandBuffer commandBuffer);

    ImageAttachment CreateImageAttachment(vk::Format format,
                                          vk::ImageUsageFlags usage,
                                          vk::ImageAspectFlags aspectFlags,
                                          vk::SampleCountFlagBits sampleCount);

    void DestroyImageAttachment(const ImageAttachment& res) {
        if (res.m_ImageView) {
            m_Device.destroyImageView(res.m_ImageView);
        }
        if (res.m_Image) {
            m_Allocator.destroyImage(res.m_Image, res.m_Allocation);
        }
    }
};
