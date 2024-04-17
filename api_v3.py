
"""
受 GPT-SoVITS 启发
"""

import os.path as osp
import re
import logging
from warnings import warn

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

import torch
from torch import nn
import torch.nn.functional as F
import librosa
import LangSegment
from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert

from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from my_utils import load_audio

import os
import sys
import traceback
from typing import Generator

from GPT_SoVITS.TTS_infer_pack.Role import RoleConfigLoader
from tools.asr.funasr_asr_utils import asr_text

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
tts_pipeline = TTS(tts_config)

APP = FastAPI()

def get_pretrain_model_path(env_name, log_file, def_path):
    """ 获取预训练模型路径
    env_name: 从环境变量获取，第一优先级
    log_file: 记录在文本文件内，第二优先级
    def_path: 传参，第三优先级
    """
    if osp.isfile(log_file):
        def_path = open(log_file, 'r', encoding="utf-8").read()
    pretrain_path = os.environ.get(env_name, def_path)
    return pretrain_path


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

gpt_path = get_pretrain_model_path('gpt_path', "./gweight.txt",
                                   "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

sovits_path = get_pretrain_model_path('sovits_path', "./sweight.txt",
                                      "GPT_SoVITS/pretrained_models/s2G488k.pth")

cnhubert_base_path = get_pretrain_model_path("cnhubert_base_path", '', "GPT_SoVITS/pretrained_models/chinese-hubert-base")

bert_path = get_pretrain_model_path("bert_path", '', "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")


is_share = eval(os.environ.get("is_share", "False"))

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

# is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
is_half = False

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

cnhubert.cnhubert_base_path = cnhubert_base_path

i18n = I18nAuto()

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)


change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {
    i18n("中文"): "all_zh"  ,  # 全部按中文识别
    i18n("英文"): "en"  ,  # 全部按英文识别#######不变
    i18n("日文"): "all_ja"  ,  # 全部按日文识别
    i18n("中英混合"): "zh"  ,  # 按中英混合识别####不变
    i18n("日英混合"): "ja"  ,  # 按日英混合识别####不变
    i18n("多语种混合"): "auto"  ,  # 多语种启动切分识别语种
}


# def clean_text_inf(text, language):
#    phones, word2ph, norm_text = clean_text(text, language)
#    phones = cleaned_text_to_sequence(phones)
#    return phones, word2ph, norm_text


def clean_text_inf(text, language):
    """
    text: 字符串
    language: 所属语言

    return:
    phones: 音素 id 序列
    word2ph: 每个字转音素后，对应的个数，对于中文，就是声韵母，因此是全是 2 的 list
    norm_text: 归一化后文本
    """
    formattext = ""
    language = language.replace("all_" ,"")
    for tmp in LangSegment.getTexts(text):
        if language == "ja":
            if tmp["lang"] == language or tmp["lang"] == "zh":
                formattext += tmp["text"] + " "
            continue
        if tmp["lang"] == language:
            formattext += tmp["text"] + " "
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    phones, word2ph, norm_text = clean_text(formattext, language)
    # print(f'音素: {phones}')
    phones = cleaned_text_to_sequence(phones)  # 统一了中、英、日等
    # print(f'音素 id: {phones}')
    return phones, word2ph, norm_text


dtype =torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language =language.replace("all_" ,"")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device  )  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, \
        {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()


@torch.no_grad()
def get_code_from_ssl(ssl):
    ssl = vq_model.ssl_proj(ssl)
    quantized, codes, commit_loss, quantized_list = vq_model.quantizer(ssl)
    # print(codes.shape, codes.dtype)  # [n_q, B, T]
    return codes.transpose(0, 1)  # [B, n_q, T]


@torch.no_grad()
def get_code_from_wav(wav_path):
    wav16k, sr = librosa.load(wav_path, sr=16000)
    if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
        # raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        warn(i18n("参考音频在3~10秒范围外，请更换！"))
    wav16k = torch.from_numpy(wav16k)
    if is_half == True:
        wav16k = wav16k.half().to(device)
    else:
        wav16k = wav16k.to(device)
    ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
        "last_hidden_state"
    ].transpose(
        1, 2
    )  # .float()
    codes = get_code_from_ssl(ssl_content)  # [B, n_q, T]

    prompt_semantic = codes[0, 0]
    return prompt_semantic


def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist) - 1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i - 1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i + 1]:
            textlist[i] += textlist[i + 1]
            del textlist[i + 1]
            del langlist[i + 1]
        else:
            i += 1

    return textlist, langlist


def nonen_clean_text_inf(text, language):
    if (language != "auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist = []
        langlist = []
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text


def get_cleaned_text_final(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja", "auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text


@torch.no_grad()
def vc_main(wav_path, text, language, prompt_wav, noise_scale=0.5):
    """ Voice Conversion
    wav_path: 待变声的源音频
    text: 对应文本
    language: 对应语言
    prompt_wav: 目标人声
    """
    language = dict_language[language]

    phones, word2ph, norm_text = get_cleaned_text_final(text, language)

    spec = get_spepc(hps, prompt_wav)
    codes = get_code_from_wav(wav_path)[None, None]  # 必须是 3D, [n_q, B, T]
    ge = vq_model.ref_enc(spec)  # [B, D, T/1]
    quantized = vq_model.quantizer.decode(codes)  # [B, D, T]
    if hps.model.semantic_frame_rate == "25hz":
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )
    _, m_p, logs_p, y_mask = vq_model.enc_p(
        quantized, torch.LongTensor([quantized.shape[-1]]),
        torch.LongTensor(phones)[None], torch.LongTensor([len(phones)]), ge
    )
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = vq_model.flow(z_p, y_mask, g=ge, reverse=True)
    o = vq_model.dec((z * y_mask)[:, :, :], g=ge)  # [B, D=1, T], torch.float32 (-1, 1)
    audio = o.detach().cpu().numpy()[0, 0]
    max_audio = np.abs(audio).max()  # 简单防止16bit爆音
    if max_audio > 1:
        audio /= max_audio
    yield hps.data.sampling_rate, (audio * 32768).astype(np.int16)


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False


class VC_Request(BaseModel):
    role: str = None
    audio_file: UploadFile


# class ROLE_Request(BaseModel):
#     t2s_weights_path: str = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
#     vits_weights_path: str = "GPT_SoVITS/pretrained_models/s2G488k.pth"
#     ref_wav: str = "/home/xfa/Documents/workspace/GPT-SoVITS/audio/ref_audio/lyx/2023_090.wav"

role_config_loader = RoleConfigLoader("roles_configs.yaml")
role_dict = role_config_loader.get_configs_for_role("default")


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if (text_lang in [None, ""]):
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "text_lang is not supported"})
    if (prompt_lang in [None, ""]):
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is not supported"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": "media_type is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400,
                            content={"message": f"text_split_method:{text_split_method} is not supported"})

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})

async def vc_handle(req: dict):
    try:
        audio_content = req.get("audio_file")
        os.makedirs("uploaded_audio", exist_ok=True)
        save_path = os.path.join("uploaded_audio", audio_content.filename)
        with open(save_path, "wb") as buffer:
            buffer.write(await audio_content.read())
        sr, audio_data = vc_main(role_dict.get("ref_wav"),role_dict.get("ref_text"),"zh",save_path)

        audio_data = pack_audio(BytesIO(), audio_data, sr, "wav").getvalue()
        return Response(audio_data, media_type=f"audio/wav")

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})

@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
        text: str = None,
        text_lang: str = None,
        ref_audio_path: str = None,
        prompt_lang: str = None,
        prompt_text: str = "",
        top_k: int = 5,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "cut0",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        streaming_mode: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.post("/svc")
async def svc_post_endpoint(vcRequest: VC_Request):
    req = vcRequest.dict()
    audio_content = req.get("audio_file")
    promote_text = asr_text(audio_content)
    tts_request = TTS_Request()
    tts_request.text_lang = 'zh'.lower()
    tts_request.prompt_lang = 'zh'
    tts_request.prompt_text = promote_text
    tts_request.ref_audio_path = role_dict.get("ref_wav")
    tts_request.text = role_dict.get("text")
    tts_request.streaming_mode = True
    return await tts_handle(tts_request)

@APP.post("/vc")
async def vc_post_endpoint(vcRequest: VC_Request):

    return await vc_handle(VC_Request)

@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


async def role_handle(role: str):
    role_dict = role_config_loader.get_configs_for_role(role)
    if role_dict:  # 假设路径一定存在
        tts_pipeline.init_vits_weights(role_dict.get("vits_weights_path"))
        tts_pipeline.init_t2s_weights(role_dict.get("t2s_weights_path"))
        tts_pipeline.set_ref_audio(role_dict.get("ref_wav"))
        JSONResponse(status_code=200, content={"message": "change role success"})
    else:
        return JSONResponse(status_code=400, content={"message": f"role not exist"})


@APP.post("/set_role")
async def role_change(role: str):
    return await role_handle(role)


if __name__ == "__main__":
    try:
        uvicorn.run(APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
