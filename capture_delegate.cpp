#include "capture_delegate.h"


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

DeckLinkCaptureDelegate::DeckLinkCaptureDelegate(BMDConfig* m_config, IDeckLinkInput* m_deckLinkInput) : 
	m_refCount(1),
	m_pixelFormat(bmdFormat8BitYUV),
	m_config(m_config),
	m_deckLinkInput(m_deckLinkInput)
{
	yadif = new Yadif(1080,1920,1920);
}



ULONG DeckLinkCaptureDelegate::AddRef(void)
{
	return __sync_add_and_fetch(&m_refCount, 1);
}

ULONG DeckLinkCaptureDelegate::Release(void)
{
	int32_t newRefValue = __sync_sub_and_fetch(&m_refCount, 1);
	if (newRefValue == 0)
	{
		delete this;
		return 0;
	}
	return newRefValue;
}



HRESULT DeckLinkCaptureDelegate::VideoInputFrameArrived(IDeckLinkVideoInputFrame* videoFrame , IDeckLinkAudioInputPacket* audioFrame )
{
	void*								frameBytes;
	// Handle Video Frame
	if (videoFrame) {
		if (videoFrame->GetFlags() & bmdFrameHasNoInputSource) {
			printf("Frame received (#%lu) - No input signal detected\n", m_frameCount);
		}
		else {
			const char *timecodeString = NULL;
			if (m_config->m_timecodeFormat != 0) {
				IDeckLinkTimecode *timecode;
				if (videoFrame->GetTimecode(m_config->m_timecodeFormat, &timecode) == S_OK) {
					timecode->GetString(&timecodeString);
				}
			}

			// printf("Frame received (#%lu) [%s] - %s - Size: %li bytes\n", m_frameCount,
			// 	timecodeString != NULL ? timecodeString : "No timecode", "Valid Frame",
			// 	videoFrame->GetRowBytes() * videoFrame->GetHeight());

			// -------------OPEN CV FRAME DISPLAY-----------------------------
			void* frameBytes;
			videoFrame->GetBytes(&frameBytes);

			unsigned char *yuyv = (unsigned char *)frameBytes;
			unsigned char *y_channel = new unsigned char[1920*1080];
			unsigned char *y_channel_de = new unsigned char[1920*1080];
			for (int i{0}; i <1920;i++) {
				for (int j{0}; j <1080;j++) {
					y_channel[i+1920*j] = (unsigned char)yuyv[1+2*i+1920*2*j];
				}
			}

			
			yadif->filter(y_channel,y_channel_de);
			cv::Mat im(videoFrame->GetHeight(), videoFrame->GetWidth(), CV_8UC1,y_channel_de);
			cv::Mat img_bgr;
			cv::cvtColor(im,img_bgr,cv::COLOR_GRAY2BGR); //3840*2160
			cv::resize(img_bgr,img_bgr,cv::Size(3840*3/4,2160*3/4));
			cv::imshow("frame",img_bgr);
			cv::waitKey(1);


			// cv::Mat im(videoFrame->GetHeight(), videoFrame->GetWidth(), CV_8UC2,frameBytes);
			// cv::Mat img_bgr;
			// cv::cvtColor(im,img_bgr,cv::COLOR_YUV2BGR_UYVY); //3840*2160
			// cv::resize(img_bgr,img_bgr,cv::Size(3840*3/4,2160*3/4));
			// cv::imshow("frame",img_bgr);
			// cv::waitKey(1);
			// -------------END OF OPEN CV FRAME DISPLAY-----------------------------
			
			if (timecodeString)
				free((void*)timecodeString);
		}

		m_frameCount++;
	}

	if (m_config->m_maxFrames > 0 && videoFrame && m_frameCount >= m_config->m_maxFrames) {
		quit = true;
		// pthread_cond_signal(&g_sleepCond);
	}

	return S_OK;
}



HRESULT DeckLinkCaptureDelegate::VideoInputFormatChanged(BMDVideoInputFormatChangedEvents events, IDeckLinkDisplayMode *mode, BMDDetectedVideoInputFormatFlags formatFlags)
{
	// This only gets called if bmdVideoInputEnableFormatDetection was set
	// when enabling video input
	HRESULT	result;
	char*	displayModeName = NULL;
	BMDPixelFormat	pixelFormat = m_pixelFormat;
	
	if (events & bmdVideoInputColorspaceChanged)
	{
		// Detected a change in colorspace, change pixel format to match detected format
		if (formatFlags & bmdDetectedVideoInputRGB444)
			pixelFormat = bmdFormat10BitRGB;
		else if (formatFlags & bmdDetectedVideoInputYCbCr422)
			pixelFormat = (m_config->m_pixelFormat == bmdFormat8BitYUV) ? bmdFormat8BitYUV : bmdFormat10BitYUV;
		else
			goto bail;
	}

	// Restart streams if either display mode or pixel format have changed
	if ((events & bmdVideoInputDisplayModeChanged) || (m_pixelFormat != pixelFormat))
	{
		mode->GetName((const char**)&displayModeName);
		printf("Video format changed to %s %s\n", displayModeName, formatFlags & bmdDetectedVideoInputRGB444 ? "RGB" : "YUV");

		if (displayModeName)
			free(displayModeName);

		if (m_deckLinkInput)
		{
			m_deckLinkInput->StopStreams();

			result = m_deckLinkInput->EnableVideoInput(mode->GetDisplayMode(), pixelFormat, m_config->m_inputFlags);
			if (result != S_OK)
			{
				fprintf(stderr, "Failed to switch video mode\n");
				goto bail;
			}

			m_deckLinkInput->StartStreams();
		}

		m_pixelFormat = pixelFormat;
	}

bail:
	return S_OK;
}
