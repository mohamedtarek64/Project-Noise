import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

def load_audio(file_path):
    """تحميل ملف صوتي وإرجاع الإشارة ومعدل العينة"""
    try:
        sample_rate, signal = wavfile.read(file_path)
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # أخذ القناة الأولى إذا كان ستيريو
        return sample_rate, signal
    except Exception as e:
        print(f"خطأ في تحميل الملف: {e}")
        return None, None

def save_audio(file_path, signal, sample_rate):
    """حفظ الإشارة كملف WAV"""
    try:
        wavfile.write(file_path, sample_rate, np.int16(signal))
        print(f"تم حفظ الملف بنجاح في: {file_path}")
    except Exception as e:
        print(f"خطأ في حفظ الملف: {e}")

def denoise_fft(signal, threshold_percent=95):
    """إزالة الضوضاء باستخدام FFT""" 
    try:
        fft_signal = np.fft.fft(signal)
        magnitudes = np.abs(fft_signal)
        threshold = np.percentile(magnitudes, threshold_percent)
        fft_signal[magnitudes < threshold] = 0
        return np.real(np.fft.ifft(fft_signal))
    except Exception as e:
        print(f"خطأ في معالجة الإشارة: {e}")
        return None

if __name__ == "__main__":
    # المسارات - تعديلها حسب موقع ملفك
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path =  os.path.join(current_dir, "untitled.wav")
    output_path = os.path.join(current_dir, "denoised_audio.wav")
    
    # 1. تحميل الملف
    sample_rate, signal = load_audio(input_path)
    
    if sample_rate is not None and signal is not None:
        # 2. إزالة الضوضاء
        denoised_signal = denoise_fft(signal)
        
        if denoised_signal is not None:
            # 3. حفظ الملف المُنقى
            save_audio(output_path, denoised_signal, sample_rate)
            
            # 4. عرض النتائج
            plt.figure(figsize=(12, 8))
            
            plt.subplot(211)
            plt.plot(signal, label='الإشارة الأصلية', color='blue')
            plt.legend()
            
            plt.subplot(212)
            plt.plot(denoised_signal, label='الإشارة المُنقاة', color='green')
            plt.legend()
            
            plt.tight_layout()
            plt.show()