import io
import os
import traceback

from starlette.responses import JSONResponse
from funasr import AutoModel
# import uvicorn
# from fastapi import FastAPI, UploadFile, File
from fastapi import UploadFile
# APP = FastAPI()


# @APP.post("/audio/")
async def asr_text(audio_files: UploadFile):
    print("get audio_files")
    if not audio_files.content_type.startswith("audio/"):
        return JSONResponse(status_code=400, content={"message": "file type is not supported"})

    os.makedirs("uploaded_audio", exist_ok=True)
    save_path = os.path.join("uploaded_audio", audio_files.filename)
    with open(save_path, "wb") as buffer:
        buffer.write(await audio_files.read())

    text = transcribe_audio(save_path)
    return text


def transcribe_audio(input_file):
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
        text = model.generate(input=input_file)[0]["text"]
    except:
        text = ''
        print(traceback.format_exc())
    return text


# if __name__ == "__main__":
#     try:
#         uvicorn.run(APP, host="0.0.0.0", port=9891)
#     except Exception as e:
#         exit(0)
