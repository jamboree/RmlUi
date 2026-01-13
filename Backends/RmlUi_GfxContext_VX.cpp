#include "RmlUi_GfxContext_VX.h"
#include <RmlUi/Core/Log.h>
#include <algorithm>

bool PhysicalDeviceInfo::Init(vx::PhysicalDevice physicalDevice) {
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

bool PhysicalDeviceInfo::HasExtension(std::string_view name) const noexcept {
    const auto it = std::ranges::lower_bound(
        m_ExtensionProperties, name, std::ranges::less{},
        [](const vk::ExtensionProperties& props) {
            return props.getExtensionName();
        });
    return it != m_ExtensionProperties.end() && it->getExtensionName() == name;
}

#define REQIRE_FEATURE(feature)                                                \
    if (!supported.feature)                                                    \
        return false;                                                          \
    feature = true

struct GfxContext_VX::DeviceFeatures
    : vx::StructureChain<
          vk::PhysicalDeviceFeatures2,
          vk::PhysicalDeviceTimelineSemaphoreFeatures,
          vk::PhysicalDeviceSynchronization2Features,
          vk::PhysicalDeviceDynamicRenderingFeatures,
          vk::PhysicalDeviceDescriptorIndexingFeatures,
          vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
          vk::PhysicalDeviceBufferDeviceAddressFeatures,
          vk::PhysicalDeviceUniformBufferStandardLayoutFeatures> {
    bool Init(vk::PhysicalDevice physicalDevice) {
        DeviceFeatures supported;
        physicalDevice.getFeatures2(&supported);
        REQIRE_FEATURE(timelineSemaphore);
        REQIRE_FEATURE(synchronization2);
        REQIRE_FEATURE(dynamicRendering);
        REQIRE_FEATURE(extendedDynamicState3ColorBlendEnable);
        REQIRE_FEATURE(extendedDynamicState3ColorBlendEquation);
        REQIRE_FEATURE(bufferDeviceAddress);
        REQIRE_FEATURE(descriptorBindingPartiallyBound);
        REQIRE_FEATURE(descriptorBindingStorageBufferUpdateAfterBind);
        REQIRE_FEATURE(descriptorBindingSampledImageUpdateAfterBind);
        REQIRE_FEATURE(uniformBufferStandardLayout);
        return true;
    }
};

void GfxContext_VX::DestroyFrameResources() {
    for (auto& frameResource : m_FrameResources) {
        if (frameResource.m_ImageView) {
            m_Device.destroyImageView(frameResource.m_ImageView);
        }
        if (frameResource.m_RenderSemaphore) {
            m_Device.destroySemaphore(frameResource.m_RenderSemaphore);
        }
    }
    DestroyImageAttachment(m_DepthStencilImage);
}

void GfxContext_VX::Destroy() {
    DestroyFrameResources();
    if (m_Swapchain) {
        m_Device.destroySwapchainKHR(m_Swapchain);
    }
    if (m_TempSemaphore) {
        m_Device.destroySemaphore(m_TempSemaphore);
    }
    for (const auto fence : m_RenderFences) {
        if (fence) {
            m_Device.destroyFence(fence);
        }
    }
    for (const auto semaphore : m_AcquireSemaphores) {
        if (semaphore) {
            m_Device.destroySemaphore(semaphore);
        }
    }
    if (m_DescriptorPool) {
        m_Device.destroyDescriptorPool(m_DescriptorPool);
    }
    if (m_TempCommandPool) {
        m_Device.destroyCommandPool(m_TempCommandPool);
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
#ifndef NDEBUG
    if (m_DebugMessenger) {
        m_Instance.destroyDebugUtilsMessengerEXT(m_DebugMessenger);
    }
#endif
    if (m_Instance) {
        m_Instance.destroy();
    }
}

bool GfxContext_VX::InitFrame() {
    const auto& acquireSemaphore = m_AcquireSemaphores[m_FrameNumber];
    m_Allocator.setCurrentFrameIndex(m_FrameNumber);
    if (auto ret = m_Device.acquireNextImageKHR(m_Swapchain, UINT64_MAX,
                                                acquireSemaphore);
        ret.result == vk::Result::eErrorOutOfDateKHR) [[unlikely]] {
        return true;
    } else {
        if (ret.result == vk::Result::eSuboptimalKHR) {
            ret.result = vk::Result::eSuccess;
            m_RenderTargetOutdated = true;
        }
        m_ImageIndex = ret.get();
    }
    return false;
}

vx::CommandBuffer GfxContext_VX::BeginFrame() {
    const vx::CommandBuffer commandBuffer{m_CommandBuffers[m_FrameNumber]};
    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::bOneTimeSubmit);
    check(commandBuffer.begin(beginInfo));
    return commandBuffer;
}

void GfxContext_VX::EndFrame() {
    const auto& acquireSemaphore = m_AcquireSemaphores[m_FrameNumber];
    const auto& renderSemaphore = CurrentFrameResource().m_RenderSemaphore;
    const auto commandBuffer = m_CommandBuffers[m_FrameNumber];

    {
        vx::ImageMemoryBarrierState imageMemoryBarrier;
        imageMemoryBarrier.init(
            CurrentFrameResource().m_Image,
            vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
        imageMemoryBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
        imageMemoryBarrier.setSrcStageAccess(
            vk::PipelineStageFlagBits2::bTransfer,
            vk::AccessFlagBits2::bTransferWrite);
        imageMemoryBarrier.setNewLayout(vk::ImageLayout::ePresentSrcKHR);
        imageMemoryBarrier.setDstStageAccess(
            vk::PipelineStageFlagBits2::bAllCommands,
            vk::AccessFlagBits2::eNone);
        commandBuffer.cmdPipelineBarriers(imageMemoryBarrier);
    }
    check(commandBuffer.end());

    vk::CommandBufferSubmitInfo bufferSubmitInfo;
    bufferSubmitInfo.setCommandBuffer(commandBuffer);
    vk::SemaphoreSubmitInfo acquireSemaphoreInfo;
    acquireSemaphoreInfo.setSemaphore(acquireSemaphore);
    acquireSemaphoreInfo.setStageMask(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput);
    vk::SemaphoreSubmitInfo renderSemaphoreInfo;
    renderSemaphoreInfo.setSemaphore(renderSemaphore);
    renderSemaphoreInfo.setStageMask(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput);
    vk::SubmitInfo2 submitInfo;
    submitInfo.setCommandBufferInfoCount(1);
    submitInfo.setCommandBufferInfos(&bufferSubmitInfo);
    submitInfo.setWaitSemaphoreInfoCount(1);
    submitInfo.setWaitSemaphoreInfos(&acquireSemaphoreInfo);
    submitInfo.setSignalSemaphoreInfoCount(1);
    submitInfo.setSignalSemaphoreInfos(&renderSemaphoreInfo);

    check(m_Device.resetFences(1, m_RenderFences + m_FrameNumber));
    check(m_Queue.submit2(1, &submitInfo, m_RenderFences[m_FrameNumber]));

    vk::PresentInfoKHR presentInfo;
    presentInfo.setSwapchainCount(1);
    presentInfo.setSwapchains(&m_Swapchain);
    presentInfo.setImageIndices(&m_ImageIndex);
    presentInfo.setWaitSemaphoreCount(1);
    presentInfo.setWaitSemaphores(&renderSemaphore);

    if (const auto ret = m_Queue.presentKHR(presentInfo);
        ret == vk::Result::eErrorOutOfDateKHR ||
        ret == vk::Result::eSuboptimalKHR) {
        m_RenderTargetOutdated = true;
        return;
    } else [[likely]] {
        check(ret);
    }

    m_FrameNumber = (m_FrameNumber + 1) % InFlightCount;
}

void GfxContext_VX::RecreateRenderTarget(vk::Extent2D extent) {
    m_FrameNumber = 0;
    (void)m_Device.waitIdle();
    DestroyFrameResources();
    m_FrameResources.count = 0;
    const auto oldSwapchain = m_Swapchain;
    InitRenderTarget(extent);
    m_Device.destroySwapchainKHR(oldSwapchain);
}

void GfxContext_VX::InitInstance(std::vector<const char*>& extensions) {
    vk::ApplicationInfo appInfo;
    appInfo.setApiVersion(VulkanApiVersion);

    vk::InstanceCreateInfo instInfo;
    instInfo.setApplicationInfo(&appInfo);

    std::initializer_list<const char*> layers{
#ifndef NDEBUG
        "VK_LAYER_KHRONOS_validation"
#endif
    };
    instInfo.setEnabledLayerCount(uint32_t(layers.size()));
    instInfo.setEnabledLayerNames(layers.begin());

#ifndef NDEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Message = vk::DebugUtilsMessageTypeFlagBitsEXT;
    vx::DebugUtilsMessengerCreateInfoEXT debugInfo;
    debugInfo.setMessageSeverity(Severity::bError);
    debugInfo.setMessageType(Message::bGeneral | Message::bValidation |
                             Message::bPerformance);
    debugInfo.setPfnUserCallback(
        [](vk::DebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
           vk::DebugUtilsMessageTypeFlagsEXT /*messageType*/,
           const vk::DebugUtilsMessengerCallbackDataEXT& callbackData,
           void* /*pUserData*/) -> VkBool32 {
            Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan error (%s): %s",
                              callbackData.pMessageIdName,
                              callbackData.pMessage);
            return false;
        });
    instInfo.attach(debugInfo);
#endif

    instInfo.setEnabledExtensionCount(uint32_t(extensions.size()));
    instInfo.setEnabledExtensionNames(extensions.data());

    m_Instance = vk::createInstance(instInfo).get();
    volkLoadInstanceOnly(m_Instance.handle);
#ifndef NDEBUG
    m_DebugMessenger = m_Instance.createDebugUtilsMessengerEXT(debugInfo).get();
#endif
}

bool GfxContext_VX::InitContext() {
    PhysicalDeviceInfo deviceInfo;
    DeviceFeatures features;
    m_PhysicalDevice = SelectPhysicalDevice(deviceInfo, features);
    if (!m_PhysicalDevice) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "no capable device");
        return false;
    }
    InitDevice(deviceInfo, features);
    InitSyncObjects();
    return true;
}

void GfxContext_VX::InitRenderTarget(vk::Extent2D extent) {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities;
    check(m_PhysicalDevice.getSurfaceCapabilitiesKHR(m_Surface,
                                                     &surfaceCapabilities));
    UpdateExtent(surfaceCapabilities, extent);
    BuildSwapchain(surfaceCapabilities);
    m_DepthStencilImage = CreateImageAttachment(
        m_DepthStencilImageFormat,
        vk::ImageUsageFlagBits::bDepthStencilAttachment,
        vk::ImageAspectFlagBits::bDepth | vk::ImageAspectFlagBits::bStencil,
        m_SampleCount);
    BuildFrameResources();
    m_FrameNumber = InFlightCount - 1;
    m_RenderTargetOutdated = false;
}

bool GfxContext_VX::QueryPhysicalDevice(vx::PhysicalDevice physicalDevice,
                                        PhysicalDeviceInfo& deviceInfo) const {
    if (!deviceInfo.Init(physicalDevice)) {
        return false;
    }
    if (deviceInfo.m_Properties.getApiVersion() < VulkanApiVersion) {
        return false;
    }
    for (const auto extension : RequiredDeviceExtensions) {
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
    return true;
}

vk::PhysicalDevice
GfxContext_VX::SelectPhysicalDevice(PhysicalDeviceInfo& deviceInfo,
                                    DeviceFeatures& features) {
    vk::PhysicalDevice firstDevice;
    PhysicalDeviceInfo firstDeviceInfo;
    for (const auto physicalDevice :
         m_Instance.enumeratePhysicalDevices().get()) {
        if (QueryPhysicalDevice(physicalDevice, deviceInfo)) {
            if (features.Init(physicalDevice)) {
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
    }
    if (firstDevice) {
        deviceInfo = std::move(firstDeviceInfo);
    }
    return firstDevice;
}

void GfxContext_VX::InitDevice(PhysicalDeviceInfo& physicalDeviceInfo,
                               vk::PhysicalDeviceFeatures2& features) {
    m_QueueFamilyIndex = FindQueueFamilyIndex(
        physicalDeviceInfo.m_QueueFamilyProperties,
        vk::QueueFlagBits::bGraphics | vk::QueueFlagBits::bTransfer, m_Surface);
    const float queuePriorities[1] = {1.0f};
    vk::DeviceQueueCreateInfo queueInfo;
    queueInfo.setQueueFamilyIndex(m_QueueFamilyIndex);
    queueInfo.setQueueCount(1);
    queueInfo.setQueuePriorities(queuePriorities);

    const std::span extensions(RequiredDeviceExtensions);
    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfoCount(1);
    deviceInfo.setQueueCreateInfos(&queueInfo);
    deviceInfo.setEnabledExtensionCount(uint32_t(extensions.size()));
    deviceInfo.setEnabledExtensionNames(extensions.data());
    deviceInfo.attachHead(features);

    m_Device = m_PhysicalDevice.createDevice(deviceInfo).get();
    volkLoadDevice(m_Device.handle);
    m_Queue = m_Device.getQueue(m_QueueFamilyIndex, 0);

    vma::AllocatorCreateInfo allocatorInfo;
    allocatorInfo.setInstance(m_Instance);
    allocatorInfo.setPhysicalDevice(m_PhysicalDevice);
    allocatorInfo.setDevice(m_Device);
    allocatorInfo.setFlags(vma::AllocatorCreateFlagBits::bBufferDeviceAddress |
                           vma::AllocatorCreateFlagBits::bMemoryBudgetEXT);
    allocatorInfo.setVulkanApiVersion(VulkanApiVersion);

    m_Allocator = vma::createAllocator(allocatorInfo).get();

    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.setQueueFamilyIndex(m_QueueFamilyIndex);
    poolInfo.setFlags(vk::CommandPoolCreateFlagBits::bResetCommandBuffer);

    m_CommandPool = m_Device.createCommandPool(poolInfo).get();

    poolInfo.setFlags(vk::CommandPoolCreateFlagBits::bTransient |
                      vk::CommandPoolCreateFlagBits::bResetCommandBuffer);
    m_TempCommandPool = m_Device.createCommandPool(poolInfo).get();

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(m_CommandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(uint32_t(std::size(m_CommandBuffers)));

    check(m_Device.allocateCommandBuffers(allocInfo, m_CommandBuffers));

    vk::DescriptorPoolSize descriptorPoolSize{
        /*type*/
        vk::DescriptorType::eCombinedImageSampler,
        /*descriptorCount*/ InFlightCount};
    vk::DescriptorPoolCreateInfo descriptorPoolInfo;
    descriptorPoolInfo.setMaxSets(1);
    descriptorPoolInfo.setFlags(
        vk::DescriptorPoolCreateFlagBits::bFreeDescriptorSet);
    descriptorPoolInfo.setPoolSizeCount(1);
    descriptorPoolInfo.setPoolSizes(&descriptorPoolSize);

    m_DescriptorPool = m_Device.createDescriptorPool(descriptorPoolInfo).get();
}

void GfxContext_VX::InitSyncObjects() {
    const vk::SemaphoreCreateInfo semaphoreInfo;
    for (auto& semaphore : m_AcquireSemaphores) {
        semaphore = m_Device.createSemaphore(semaphoreInfo).get();
    }
    vk::FenceCreateInfo fenceInfo;
    fenceInfo.setFlags(vk::FenceCreateFlagBits::bSignaled);
    for (auto& fence : m_RenderFences) {
        fence = m_Device.createFence(fenceInfo).get();
    }
    m_TempSemaphore = m_Device.createTimelineSemaphore(1).get();
}

void GfxContext_VX::BuildSwapchain(
    const vk::SurfaceCapabilitiesKHR& capabilities) {
    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.setSurface(m_Surface);
    swapchainInfo.setMinImageCount(std::clamp(
        3u, capabilities.getMinImageCount(), capabilities.getMaxImageCount()));
    swapchainInfo.setImageFormat(m_SwapchainImageFormat);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent(m_FrameExtent);
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::bColorAttachment |
                                vk::ImageUsageFlagBits::bTransferDst);
    swapchainInfo.setImageSharingMode(vk::SharingMode::eExclusive);
    swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::bIdentity);
    swapchainInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::bOpaque);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
    swapchainInfo.setClipped(true);
    swapchainInfo.setOldSwapchain(m_Swapchain);

    m_Swapchain = m_Device.createSwapchainKHR(swapchainInfo).get();
}

ImageAttachment GfxContext_VX::CreateImageAttachment(
    vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageAspectFlags aspectFlags, vk::SampleCountFlagBits sampleCount) {
    auto imageInfo = vx::image2DCreateInfo(format, m_FrameExtent, usage);
    imageInfo.setSamples(sampleCount);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setFlags(vma::AllocationCreateFlagBits::bDedicatedMemory);
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    ImageAttachment res;
    res.m_Image =
        m_Allocator.createImage(imageInfo, allocationInfo, &res.m_Allocation)
            .get();

    const auto imageViewInfo =
        vx::imageViewCreateInfo(vk::ImageViewType::e2D, res.m_Image, format,
                                vx::subresourceRange(aspectFlags));
    res.m_ImageView = m_Device.createImageView(imageViewInfo).get();
    return res;
}

void GfxContext_VX::BuildFrameResources() {
    const auto swapchainImages =
        m_Device.getSwapchainImagesKHR(m_Swapchain).get();
    m_FrameResources.count = swapchainImages.count;
    m_FrameResources.prepare();
    const vk::SemaphoreCreateInfo semaphoreInfo;

    for (unsigned i = 0; i != swapchainImages.count; ++i) {
        auto& frameResource = m_FrameResources[i];
        frameResource.m_Image = swapchainImages[i];
        const auto imageViewInfo = vx::imageViewCreateInfo(
            vk::ImageViewType::e2D, frameResource.m_Image,
            m_SwapchainImageFormat,
            vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
        frameResource.m_ImageView =
            m_Device.createImageView(imageViewInfo).get();
        frameResource.m_RenderSemaphore =
            m_Device.createSemaphore(semaphoreInfo).get();
    }
}

void GfxContext_VX::UpdateExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                                 vk::Extent2D extent) {
    if (capabilities.currentExtent.width == ~0u) {
        const auto minExtent = capabilities.minImageExtent;
        const auto maxExtent = capabilities.maxImageExtent;
        m_FrameExtent.width =
            std::clamp(extent.width, minExtent.width, maxExtent.width);
        m_FrameExtent.height =
            std::clamp(extent.height, minExtent.height, maxExtent.height);
    } else [[likely]] {
        m_FrameExtent = capabilities.getCurrentExtent();
    }
}

uint32_t GfxContext_VX::FindQueueFamilyIndex(
    const vx::List<vk::QueueFamilyProperties>& queueFamilyProperties,
    vk::QueueFlags flags, vk::SurfaceKHR surface) const {
    unsigned extraBitCountMin = ~0u;
    uint32_t index = ~0u;
    for (uint32_t i = 0; i != queueFamilyProperties.count; ++i) {
        if (surface) {
            const auto val = m_PhysicalDevice.getSurfaceSupportKHR(i, surface);
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

vx::CommandBuffer GfxContext_VX::BeginTemp() {
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(m_TempCommandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    vk::CommandBuffer commandBuffer;
    check(m_Device.allocateCommandBuffers(allocInfo, &commandBuffer));

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::bOneTimeSubmit);
    check(commandBuffer.begin(beginInfo));
    return commandBuffer;
}

void GfxContext_VX::EndTemp(vk::CommandBuffer commandBuffer) {
    check(commandBuffer.end());

    vk::CommandBufferSubmitInfo bufferSubmitInfo;
    bufferSubmitInfo.setCommandBuffer(commandBuffer);

    vk::SubmitInfo2 submitInfo;
    submitInfo.setCommandBufferInfoCount(1);
    submitInfo.setCommandBufferInfos(&bufferSubmitInfo);

    const auto waitValue =
        m_Device.getSemaphoreCounterValue(m_TempSemaphore).get() + 1;
    vk::SemaphoreSubmitInfo signalSemaphoreInfo;
    signalSemaphoreInfo.setSemaphore(m_TempSemaphore);
    signalSemaphoreInfo.setStageMask(vk::PipelineStageFlagBits2::bAllCommands);
    signalSemaphoreInfo.setValue(waitValue);
    submitInfo.setSignalSemaphoreInfoCount(1);
    submitInfo.setSignalSemaphoreInfos(&signalSemaphoreInfo);

    check(m_Queue.submit2(1, &submitInfo));

    vk::SemaphoreWaitInfo waitInfo;
    waitInfo.setSemaphoreCount(1);
    waitInfo.setSemaphores(&m_TempSemaphore);
    waitInfo.setValues(&waitValue);
    check(m_Device.waitSemaphores(waitInfo, UINT64_MAX));

    m_Device.freeCommandBuffers(m_TempCommandPool, 1, &commandBuffer);
}
