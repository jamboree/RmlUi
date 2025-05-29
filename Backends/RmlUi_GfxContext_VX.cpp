#include "RmlUi_GfxContext_VX.h"
#include <RmlUi/Core/Log.h>

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

struct GfxContext_VX::DeviceFeatures
    : vx::StructureChain<
          vk::PhysicalDeviceFeatures2,
          vk::PhysicalDeviceTimelineSemaphoreFeatures,
          vk::PhysicalDeviceSynchronization2Features,
          vk::PhysicalDeviceDynamicRenderingFeatures,
          vk::PhysicalDeviceBufferDeviceAddressFeatures,
          vk::PhysicalDeviceUniformBufferStandardLayoutFeatures> {
    bool Init(vk::PhysicalDevice physicalDevice) {
        DeviceFeatures supported;
        physicalDevice.getFeatures2(&supported);
        if (!supported.timelineSemaphore)
            return false;
        timelineSemaphore = true;
        if (!supported.synchronization2)
            return false;
        synchronization2 = true;
        if (!supported.dynamicRendering)
            return false;
        dynamicRendering = true;
        if (!supported.bufferDeviceAddress)
            return false;
        bufferDeviceAddress = true;
        if (!supported.uniformBufferStandardLayout)
            return false;
        uniformBufferStandardLayout = true;
        return true;
    }
};

void GfxContext_VX::DestroyFrameResources() {
    for (auto& frameResource : m_FrameResources) {
        if (frameResource.m_ImageView) {
            m_Device.destroyImageView(frameResource.m_ImageView);
        }
    }
}

void GfxContext_VX::Destroy() {
    for (uint32_t i = 0; i != InFlightCount; ++i) {
        m_Renderer.ResetFrame(i);
    }
    m_Renderer.Shutdown();

    DestroyFrameResources();
    DestroyImageAttachment(m_DepthStencilImage);
    if (m_Swapchain) {
        m_Device.destroySwapchainKHR(m_Swapchain);
    }
    if (m_TempSemaphore) {
        m_Device.destroySemaphore(m_TempSemaphore);
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

void GfxContext_VX::BeginFrame(vk::Extent2D extent) {
    const auto& syncObject = m_SyncObjects[m_FrameNumber];
    check(
        m_Device.waitForFences(1, &syncObject.m_RenderFence, true, UINT64_MAX));
    m_Renderer.ResetFrame(m_FrameNumber);
    m_Allocator.setCurrentFrameIndex(m_FrameNumber);
    if (auto ret = m_Device.acquireNextImageKHR(m_Swapchain, UINT64_MAX,
                                                syncObject.m_AcquireSemaphore);
        ret.result == vk::Result::eErrorOutOfDateKHR) [[unlikely]] {
        RecreateRenderTarget(extent);
    } else {
        if (ret.result == vk::Result::eSuboptimalKHR) {
            ret.result = vk::Result::eSuccess;
            m_RenderTargetOutdated = true;
        }
        m_ImageIndex = ret.get();
    }
    const vx::CommandBuffer commandBuffer{m_CommandBuffers[m_FrameNumber]};
    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::bOneTimeSubmit);
    check(commandBuffer.begin(beginInfo));

    enum { ColorImage, DepthStencilImage, ImageCount };
    vx::ImageMemoryBarrierState imageMemoryBarriers[ImageCount] = {
        {m_FrameResources[m_ImageIndex].m_Image,
         vx::allSubresourceRange(vk::ImageAspectFlagBits::bColor)},
        {m_DepthStencilImage.m_Image,
         vx::allSubresourceRange(vk::ImageAspectFlagBits::bDepth |
                                 vk::ImageAspectFlagBits::bStencil)},
    };

    imageMemoryBarriers[ColorImage].update(
        vk::ImageLayout::eAttachmentOptimal,
        vk::PipelineStageFlagBits2::bColorAttachmentOutput,
        vk::AccessFlagBits2::bColorAttachmentWrite);
    imageMemoryBarriers[DepthStencilImage].setSrcStageAccess(
        vk::PipelineStageFlagBits2::bLateFragmentTests,
        vk::AccessFlagBits2::bDepthStencilAttachmentWrite);
    imageMemoryBarriers[DepthStencilImage].setNewLayout(
        vk::ImageLayout::eAttachmentOptimal);
    imageMemoryBarriers[DepthStencilImage].setDstStageAccess(
        vk::PipelineStageFlagBits2::bEarlyFragmentTests |
            vk::PipelineStageFlagBits2::bLateFragmentTests,
        vk::AccessFlagBits2::bDepthStencilAttachmentRead |
            vk::AccessFlagBits2::bDepthStencilAttachmentWrite);
    commandBuffer.cmdPipelineBarriers(rawIns(imageMemoryBarriers));

    vk::RenderingAttachmentInfo colorAttachmentInfo;
    colorAttachmentInfo.setImageLayout(
        imageMemoryBarriers[ColorImage].getNewLayout());
    colorAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachmentInfo.setImageView(
        m_FrameResources[m_ImageIndex].m_ImageView);
    colorAttachmentInfo.setClearValue({.color = vk::ClearColorValue()});

    vk::RenderingAttachmentInfo depthStencilAttachmentInfo;
    depthStencilAttachmentInfo.setImageLayout(
        imageMemoryBarriers[DepthStencilImage].getNewLayout());
    depthStencilAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
    depthStencilAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eDontCare);
    depthStencilAttachmentInfo.setImageView(m_DepthStencilImage.m_ImageView);
    depthStencilAttachmentInfo.setClearValue(
        {.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}});

    vk::RenderingInfo renderingInfo;
    renderingInfo.setRenderArea({{0, 0}, m_FrameExtent});
    renderingInfo.setLayerCount(1);
    renderingInfo.setColorAttachmentCount(1);
    renderingInfo.setColorAttachments(&colorAttachmentInfo);
    renderingInfo.setDepthAttachment(&depthStencilAttachmentInfo);
    renderingInfo.setStencilAttachment(&depthStencilAttachmentInfo);

    commandBuffer.cmdBeginRendering(renderingInfo);

    vk::Viewport viewport;
    viewport.setWidth(float(m_FrameExtent.getWidth()));
    viewport.setHeight(float(m_FrameExtent.getHeight()));
    viewport.setMinDepth(0.f);
    viewport.setMaxDepth(1.f);
    commandBuffer.cmdSetViewport(0, 1, &viewport);
    m_Renderer.BeginFrame(commandBuffer, m_FrameNumber);
}

void GfxContext_VX::EndFrame() {
    const auto& syncObject = m_SyncObjects[m_FrameNumber];
    const auto commandBuffer = m_CommandBuffers[m_FrameNumber];

    m_Renderer.EndFrame();
    commandBuffer.cmdEndRendering();
    {
        vx::ImageMemoryBarrierState imageMemoryBarrier{
            m_FrameResources[m_ImageIndex].m_Image,
            vx::allSubresourceRange(vk::ImageAspectFlagBits::bColor)};
        imageMemoryBarrier.setOldLayout(vk::ImageLayout::eAttachmentOptimal);
        imageMemoryBarrier.setSrcStageAccess(
            vk::PipelineStageFlagBits2::bColorAttachmentOutput,
            vk::AccessFlagBits2::bColorAttachmentWrite);
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
    acquireSemaphoreInfo.setSemaphore(syncObject.m_AcquireSemaphore);
    acquireSemaphoreInfo.setStageMask(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput);
    vk::SemaphoreSubmitInfo renderSemaphoreInfo;
    renderSemaphoreInfo.setSemaphore(syncObject.m_RenderSemaphore);
    renderSemaphoreInfo.setStageMask(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput);
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
    DestroyImageAttachment(m_DepthStencilImage);
    m_FrameResources.count = 0;
    vk::SurfaceCapabilitiesKHR surfaceCapabilities;
    check(m_PhysicalDevice.getSurfaceCapabilitiesKHR(m_Surface,
                                                     &surfaceCapabilities));
    UpdateExtent(surfaceCapabilities, extent);
    const auto oldSwapchain = m_Swapchain;
    BuildSwapchain(surfaceCapabilities);
    m_Device.destroySwapchainKHR(oldSwapchain);
    BuildDepthStencilImage();
    BuildFrameResources();
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

    vk::PipelineRenderingCreateInfo renderingInfo;
    renderingInfo.setColorAttachmentCount(1);
    renderingInfo.setColorAttachmentFormats(&m_SwapchainImageFormat);
    renderingInfo.setDepthAttachmentFormat(m_DepthStencilImageFormat);
    renderingInfo.setStencilAttachmentFormat(m_DepthStencilImageFormat);

    if (!m_Renderer.Init(*this, renderingInfo)) {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "Failed to initialize Vulkan render interface");
        return false;
    }

    return true;
}

void GfxContext_VX::InitRenderTarget(vk::Extent2D extent) {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities;
    check(m_PhysicalDevice.getSurfaceCapabilitiesKHR(m_Surface,
                                                     &surfaceCapabilities));
    UpdateExtent(surfaceCapabilities, extent);
    BuildSwapchain(surfaceCapabilities);
    BuildDepthStencilImage();
    BuildFrameResources();
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

void GfxContext_VX::BuildDepthStencilImage() {
    const auto depthImageInfo =
        vx::image2DCreateInfo(m_DepthStencilImageFormat, m_FrameExtent,
                              vk::ImageUsageFlagBits::bDepthStencilAttachment);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    m_DepthStencilImage.m_Image =
        m_Allocator
            .createImage(depthImageInfo, allocationInfo,
                         &m_DepthStencilImage.m_Allocation)
            .get();
    const auto imageViewInfo = vx::imageViewCreateInfo(
        vk::ImageViewType::e2D, m_DepthStencilImage.m_Image,
        m_DepthStencilImageFormat,
        vk::ImageAspectFlagBits::bDepth | vk::ImageAspectFlagBits::bStencil);
    m_DepthStencilImage.m_ImageView =
        m_Device.createImageView(imageViewInfo).get();
}

void GfxContext_VX::BuildFrameResources() {
    const auto swapchainImages =
        m_Device.getSwapchainImagesKHR(m_Swapchain).get();
    m_FrameResources.count = swapchainImages.count;
    m_FrameResources.prepare();

    for (unsigned i = 0; i != swapchainImages.count; ++i) {
        auto& frameResource = m_FrameResources[i];
        frameResource.m_Image = swapchainImages[i];
        const auto imageViewInfo = vx::imageViewCreateInfo(
            vk::ImageViewType::e2D, frameResource.m_Image,
            m_SwapchainImageFormat, vk::ImageAspectFlagBits::bColor);
        frameResource.m_ImageView =
            m_Device.createImageView(imageViewInfo).get();
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
