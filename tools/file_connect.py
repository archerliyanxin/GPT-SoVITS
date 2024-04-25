import os
import wave
import zipfile


def extract_wav_files(zip_file, extract_dir):
    # 解压缩压缩包中的所有wav文件到指定目录
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def concatenate_wav_files(input_dir, output_file):
    # 获取目录中所有wav文件的文件名
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    if not wav_files:
        print("No WAV files found in the directory.")
        return

    # 打开第一个wav文件以获取参数
    with wave.open(os.path.join(input_dir, wav_files[0]), 'rb') as first_wav:
        params = first_wav.getparams()

    # 创建输出wav文件
    with wave.open(output_file, 'wb') as output_wav:
        output_wav.setparams(params)

        # 逐个读取并写入输入wav文件的内容到输出文件
        for wav_file in wav_files:
            with wave.open(os.path.join(input_dir, wav_file), 'rb') as input_wav:
                output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))

    print("Concatenation complete. Output file:", output_file)




if __name__ == "__main__":
    # 设置输入压缩包和输出文件名
    zip_file = 'input.zip'
    output_file = 'output.wav'
    # 设置解压缩目录
    extract_dir = 'extracted_files'

    # 解压缩压缩包中的wav文件到指定目录
    extract_wav_files(zip_file, extract_dir)

    # 将解压缩后的多个wav文件拼接成一个大的wav文件
    concatenate_wav_files(extract_dir, output_file)

    # 可选：在完成后删除解压缩的文件夹及其中的文件
    os.rmdir(extract_dir)
