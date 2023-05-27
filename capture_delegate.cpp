#include "capture_delegate.h"


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

DeckLinkCaptureDelegate::DeckLinkCaptureDelegate(BMDConfig* m_config, IDeckLinkInput* m_deckLinkInput) : 
	m_refCount(1),
	m_pixelFormat(bmdFormat8BitYUV),
	m_config(m_config),
	m_deckLinkInput(m_deckLinkInput)
{
	yadif = new Yadif(1080,1920,1920*2);
	
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
	// printf("*****Format %x\n",videoFrame->GetPixelFormat());
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

			// printf("Frame received (#%lu) [%s] - %s - Size: %li bytes,\n", m_frameCount,
			// 	timecodeString != NULL ? timecodeString : "No timecode", "Valid Frame",
			// 	videoFrame->GetRowBytes() * videoFrame->GetHeight());

			// printf("%li,%li\n",videoFrame->GetRowBytes(),videoFrame->GetHeight());

			// -------------OPEN CV FRAME DISPLAY-----------------------------
			void* frameBytes;
			videoFrame->GetBytes(&frameBytes);
			// int width = 720;
			// int hight = 576;
			unsigned char *yuyv = (unsigned char *)frameBytes;

			
			// #TODO delete this memories
			// unsigned char *y_channel = new unsigned char[videoFrame->GetWidth()*videoFrame->GetHeight()];
			// unsigned char *u_channel = new unsigned char[videoFrame->GetWidth()*videoFrame->GetHeight()];
			// unsigned char *v_channel = new unsigned char[videoFrame->GetWidth()*videoFrame->GetHeight()];

			unsigned char *yuv_de = new unsigned char[videoFrame->GetWidth()*videoFrame->GetHeight()*2];

			// unsigned char *y_channel_de = new unsigned char[videoFrame->GetWidth()*videoFrame->GetHeight()];

			// unsigned char *u_channel_de = new unsigned char[videoFrame->GetWidth()/2*videoFrame->GetHeight()];
			// unsigned char *v_channel_de = new unsigned char[videoFrame->GetWidth()/2*videoFrame->GetHeight()];

			// if (bmdFormat8BitYUV == videoFrame->GetPixelFormat() )
				// printf("Format %x\n",videoFrame->GetPixelFormat());

			// for (int i{0}; i <videoFrame->GetWidth();i++) {
			// 	for (int j{0}; j <videoFrame->GetHeight();j++) {
			// 		y_channel[i+videoFrame->GetWidth()*j] = (unsigned char)yuyv[1+2*i+videoFrame->GetWidth()*2*j];
			// 	}
			// }

			// cv::Mat im(videoFrame->GetHeight(), videoFrame->GetWidth(), CV_8UC1,y_channel);
			// cv::Mat img_bgr;
			// cv::cvtColor(im,img_bgr,cv::COLOR_GRAY2BGR); //3840*2160
			// cv::resize(img_bgr,img_bgr,cv::Size(1920,1080));
			// cv::imshow("frame",img_bgr);
			// cv::waitKey(1);

			// 
			// yadif->filter(yuyv,yuv_de);
			// cv::Mat im(videoFrame->GetHeight(), videoFrame->GetWidth(), CV_8UC2,yuv_de);
			// yadif->filter(yuyv,yuyv);
			yadif->filter(yuyv);
			cv::Mat im(videoFrame->GetHeight(), videoFrame->GetWidth(), CV_8UC2,yuyv);
			cv::Mat img_bgr;
			cv::cvtColor(im,img_bgr,cv::COLOR_YUV2BGR_UYVY); //3840*2160
			// cv::resize(img_bgr,img_bgr,cv::Size(3840*3/4,2160*3/4));
			cv::resize(img_bgr,img_bgr,cv::Size(1920,1080));
			cv::imshow("frame",img_bgr);
			cv::waitKey(1);			

			delete[] yuv_de;

			// deinterlace y channel onlu
			// yadif->filter(yuyv,y_channel_de);
			// cv::Mat im(videoFrame->GetHeight(), videoFrame->GetWidth(), CV_8UC1,y_channel_de);
			// cv::Mat img_bgr;
			// cv::cvtColor(im,img_bgr,cv::COLOR_GRAY2BGR); //3840*2160
			// // cv::resize(img_bgr,img_bgr,cv::Size(3840*3/4,2160*3/4));
			// cv::resize(img_bgr,img_bgr,cv::Size(1920,1080));
			// cv::imshow("frame",img_bgr);
			// cv::waitKey(1);

			// display orginal frmae 	
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
		// printf("Format %x\n",videoFrame->GetPixelFormat());
	}

bail:
	return S_OK;
}
