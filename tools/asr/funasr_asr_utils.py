import io
import os
import traceback

from starlette.responses import JSONResponse
from funasr import AutoModel
# import uvicorn
# from fastapi import FastAPI, UploadFile, File
from fastapi import UploadFile
from faster_whisper import WhisperModel
# APP = FastAPI()


# @APP.post("/audio/")
async def asr_text(audio_files: UploadFile):
    print("get audio_files")
    if not audio_files.content_type.startswith("audio/"):
        raise ''

    os.makedirs("uploaded_audio", exist_ok=True)
    save_path = os.path.join("uploaded_audio", audio_files.filename)
    with open(save_path, "wb") as buffer:
        buffer.write(await audio_files.read())

    # text = transcribe_audio(save_path)
    for text_chunk in transcribe_audio_funasr(save_path):
        yield text_chunk

def transcribe_audio_funasr(input_file):
    print(f"input file %s"%(input_file))
    path_asr = 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
    path_vad = 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch'
    path_punc = 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
    path_asr = path_asr if os.path.exists(
        path_asr) else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    path_vad = path_vad if os.path.exists(path_vad) else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    path_punc = path_punc if os.path.exists(path_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

    model = AutoModel(
        model=path_asr,
        model_revision="v2.0.4",
        vad_model=path_vad,
        vad_model_revision="v2.0.4",
        punc_model=path_punc,
        punc_model_revision="v2.0.4",
    )

    try:
        yield model.generate(input=input_file)[0]["text"]
    except:
        return ''

def transcribe_audio(input_file):
    print(f"input file %s"%(input_file))

    chunk_size = [0, 8, 4]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
    encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
    decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

    model = AutoModel(
        model="tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")


    import soundfile

    wav_file = input_file
    speech, sample_rate = soundfile.read(wav_file)
    chunk_stride = chunk_size[1] * 960  # 600ms

    cache = {}
    total_chunk_num = int(len((speech) - 1) / chunk_stride)
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num
        res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
                             encoder_chunk_look_back=encoder_chunk_look_back,
                             decoder_chunk_look_back=decoder_chunk_look_back)[0].get('text')
        yield res


# if __name__ == "__main__":
#     try:
#         uvicorn.run(APP, host="0.0.0.0", port=9891)
#     except Exception as e:
#         exit(0)
