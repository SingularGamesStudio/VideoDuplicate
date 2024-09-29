import os
import random
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from ipywidgets import Video
import os
import matplotlib.pyplot as plt
import cv2

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy import signal as sig
import matplotlib.mlab as mlab

HEIGHT = 320
WIDTH = 180
VOTE_MAXDIFF = 5
CROP = 0.1
VID_TRESHOLD = 0.1
VID_SOUND_TRESHOLD = 0.2
DEFAULT_FS = 44100
DEFAULT_WINDOW_SIZE = 4096 * 4
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_MAX_FREQ = 400
PEAK_RATIO = 0.85
DERIVATIVE_TAILS = 0.05
DERIVATIVE_RATIO = 2.75

#######################################################################################################
app = FastAPI()
data = dict()
sound_data = dict()

# Директории для хранения эмбеддингов
AUDIO_EMBEDDINGS_DIR = "embeddings_audio"
VIDEO_EMBEDDINGS_DIR = "embeddings_video"


class VideoLinkRequest(BaseModel):
    name: str


def load(file):
    """
    Загрузка видеофайла
    """
    vid = cv2.VideoCapture(file)

    frames = np.zeros(
        (int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), HEIGHT, WIDTH, 3),
        dtype=np.uint8,
    )
    i = 0
    while True:
        success, frame = vid.read()
        if success:
            frames[i] = cv2.resize(frame, (WIDTH, HEIGHT))
        else:
            break
        i += 1
    return frames


def static_mask(frames):
    """
    Возвращает маску, фильтрующую статичные писклеи на видео
    """
    kmax, kmin = int(frames.shape[0] * (1 - CROP)), int(frames.shape[0] * CROP)
    diff = (
        np.max(frames[kmin:kmax], axis=0) - np.min(frames[kmin:kmax], axis=0)
    ).sum(axis=2)
    return np.less(diff, 5)


def gkern(l=5, sig=1.0):
    """
    Ядро Гаусса (сглаживание)
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / np.sum(gauss)


def get_bright(frames):
    """
    Вычисляет среднее значение яркости кадров, и возвращает сглаженную последовательность
    """
    mask = static_mask(frames)
    frames[:, mask, :] = 0
    res = np.dot(
        np.sum(np.sum(frames, axis=1), axis=1)
        / (frames.shape[1] * frames.shape[2] - mask.sum()),
        np.array([0.299, 0.587, 0.114]),
    )
    return np.convolve(res, gkern())


def get_embed(frames):
    """
    Возвращает эмбеддинг видео - производные яркости по времени по четвертям экрана
    """
    q = []
    q.append(frames[:, : HEIGHT // 2, : WIDTH // 2, :])
    q.append(frames[:, : HEIGHT // 2, WIDTH // 2 :, :])
    q.append(frames[:, HEIGHT // 2 :, : WIDTH // 2, :])
    q.append(frames[:, HEIGHT // 2 :, WIDTH // 2 :, :])
    res = []
    for i in range(4):
        bright = get_bright(q[i])
        diff = bright[1:] - bright[:-1]
        res.append(diff[int(len(diff) * CROP) : int(len(diff) * (1 - CROP))])
    return np.array(res)


def get_shift(embed1, embed2):
    """
    Возвращает сдвиг, на который нужно сдвинуть эмбеддинг 2, чтобы совпасть с эмбеддингом 1
    """
    res = np.zeros(4)
    for i in range(4):
        e12 = embed1[i] * embed1[i]
        conv = np.convolve(
            embed1[i], np.flip(embed2[i]), mode="valid"
        ) / np.sqrt(np.convolve(e12, np.ones(embed2[i].shape[0]), mode="valid"))
        res[i] = np.argmax(conv)
    return res


def vote(shift):
    """
    Голосование за значение, требуется хотя бы 3 голоса за близкие на VOTE_MAXDIFF значения
    """
    if shift.max() - shift.min() <= VOTE_MAXDIFF:
        return int(shift.mean())
    for i in range(shift.shape[0]):
        other = shift[np.arange(shift.shape[0]) != i]
        if other.max() - other.min() <= VOTE_MAXDIFF:
            return int(other.mean())
    return -1


def similarity_score(sample, base):
    """
    Возвращает число, кореллирующее с тем, насколько различны эмбеддинги (0 при одинаковых, inf при совсем различных)

    Сравниваются эмбеддинги четвертей экрана, и голосуют за сдвиг по времени, затем считается MAE
    """
    base1 = np.copy(base)
    if len(base1[0]) < len(sample[0]) - 10:
        return np.inf
    if len(base1[0]) < len(sample[0]):
        base1 = np.concatenate(
            (base1, np.zeros((4, len(sample[0]) - len(base1[0])))), axis=1
        )
    sample_norm = (sample - np.mean(sample, axis=1, keepdims=True)) / np.std(
        sample, axis=1, keepdims=True
    )
    shift = vote(get_shift(base1, sample))
    if shift == -1:
        return np.inf

    shifted = base1[:, shift : shift + len(sample[0])]

    shifted_norm = (shifted - np.mean(shifted, axis=1, keepdims=True)) / np.std(
        shifted, axis=1, keepdims=True
    )
    res = np.abs(shifted_norm - sample_norm).mean(axis=1)
    return (res.sum() - res.min() - res.max()) / 2


def similarity_scores(sample, data):
    order = data.keys()
    scores = dict()
    for key in order:
        scores[key] = similarity_score(sample, data[key])
    return scores


def extract_audio_from_video(video_path):
    """
    Извлечение аудиодорожки из видеофайла.

    Аргументы:
    - video_path: путь к видеофайлу.

    Возвращает:
    - аудиоданные.
    """
    # Используем moviepy для извлечения аудио
    video = VideoFileClip(video_path, verbose=False)
    # Сохраняем аудиодорожку в формате wav
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(
        audio_path,
        codec="pcm_s16le",
        verbose=False,
        ffmpeg_params=["-f", "wav"],
        logger=None,
    )

    audio_data, _ = librosa.load(audio_path, sr=DEFAULT_FS)
    return audio_data


def max_pooling(arr2D, K=4, L=1):
    """
    функция для пулинга

    Аргументы:
    - arr2D: двумерный массив
    - K: размер окна по вертикали
    - L: размер окна по горизонтали

    Возвращает:
    - пуллированный массив
    """
    M, N = arr2D.shape
    MK = M // K
    NL = N // L
    return arr2D[: MK * K, : NL * L].reshape(MK, K, NL, L).max(axis=(1, 3))


def get_spectrogram(audio):
    """
    функция для получения спектрограммы

    Аргументы:
    - audio: аудиоданные

    Возвращает:
    - сжатая спектрограмма
    """
    audio = np.trim_zeros(audio, "fb")  # Remove leading zeros from audio
    arr2D = mlab.specgram(
        audio,
        NFFT=DEFAULT_WINDOW_SIZE,
        Fs=DEFAULT_FS,
        window=mlab.window_hanning,
        noverlap=int(DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO),
    )[0]
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))
    return max_pooling(arr2D[:DEFAULT_MAX_FREQ, :], 4, 1)


def convolve_spectrograms(spectrogram1, spectrogram2):
    """
    Конволюция двух спектрограмм.

    Аргументы:
    - spectrogram1: первая спектрограмма.
    - spectrogram2: вторая спектрограмма.

    Возвращает:
    - спектрограмма результата конволюции.
    """
    if spectrogram1.shape[0] == 0:
        return np.zeros((0))
    if spectrogram2.shape[0] == 0:
        return np.zeros((0))
    a = np.ones(
        (spectrogram1.shape[0], spectrogram2.shape[1] - 1)
    ) * spectrogram1.mean(axis=1).reshape(-1, 1)
    b = np.ones(
        (spectrogram1.shape[0], spectrogram2.shape[1] - 1)
    ) * spectrogram1.mean(axis=1).reshape(-1, 1)
    spectrogram1 = np.concatenate([a, spectrogram1, b], axis=1)
    convolved = sig.correlate2d(
        spectrogram1,
        spectrogram2 / (spectrogram2**2).sum(axis=1).reshape(-1, 1) ** 0.5,
        mode="valid",
    )
    e12 = spectrogram1 * spectrogram1
    convolved_norm = convolved / np.sqrt(
        sig.correlate2d(e12, np.ones(spectrogram2.shape), mode="valid")
    )
    result = convolved_norm.ravel()

    return result


def check_derivative_stats(deriv):
    """
    Проверка статистик производной.

    Аргументы:
    - deriv: производная одномерного массива.

    Возвращает:
    - проверку того что производная не является шумом.
    """
    return (
        np.quantile(deriv, 1 - DERIVATIVE_TAILS)
        - np.quantile(deriv, DERIVATIVE_TAILS)
    ) * DERIVATIVE_RATIO < deriv.max() - deriv.min()


def big_delta_detector(deriv):
    """
    Поиск перепада в производной массива.

    Аргументы:
    - deriv: производная одномерного массива.

    Возвращает:
    - наличие пика.
    """
    a = min(deriv.argmax(), deriv.argmin())
    b = max(deriv.argmax(), deriv.argmin())
    if b - a <= 1:
        return True
    # print((deriv[1:]-deriv[:-1])[a+1:b], a,b)
    return (deriv[1:] - deriv[:-1])[a + 1 : b].min() > 0


def big_spike_detector(arr):
    """
    Поиск большого пика в одномерном массиве.

    Аргументы:
    - arr: одномерный массив.

    Возвращает:
    - наличие пика.
    """
    if arr.size == 0:
        return False
    max_loc = np.argmax(arr)
    left_border = max_loc - 1
    right_border = max_loc + 1
    while left_border > 0:
        if arr[left_border] > arr[left_border + 1]:
            break
        left_border -= 1
    while right_border < len(arr):
        if arr[right_border] > arr[right_border - 1]:
            break
        right_border += 1
    if right_border >= len(arr) or left_border <= 0:
        return False
    return (
        max(arr[right_border:].max(), arr[: left_border + 1].max())
        < (arr[max_loc] - arr.mean()) * PEAK_RATIO + arr.mean()
    )


def fft_dublicate_detector(arr):
    """
    Проверка совпадения двух аудиотреков по свертке их сжатых спектрограмм.

    Аргументы:
    - arr: одномерный массив. Результат конволюции двух спектрограмм.

    Возвращает:
    - наличие дубликата.
    """
    # если производная это шум, то это точно не дубликат
    deriv = np.diff(arr)
    if check_derivative_stats(np.diff(arr)) == False:
        return False
    deriv_rev = deriv[::-1]
    # проверяем наличие перепада в производной
    if (
        big_delta_detector(deriv) == True
        or big_delta_detector(deriv_rev) == True
    ):
        return True
    # проверяем наличие большого пика в исходном массиве
    return big_spike_detector(arr)


def load_sound_embeddings(embeddings_folder):
    """
    Загрузка эмбеддингов аудиодорожек.

    Аргументы:
    - embeddings_folder: путь к папке с эмбеддингами.

    Возвращает:
    - словарь эмбеддингов.
    """
    embeddings = {}
    for filename in os.listdir(embeddings_folder):
        if filename.startswith("audio_emb_"):
            id = filename[10:-4]
            embeddings[id] = np.load(f"{embeddings_folder}/{filename}")
    return embeddings


def similar_sound(duplicate_emb, parent_emb):
    """
    Возвращает, совпадают ли звуковые эмбеддинги
    """
    convolved = convolve_spectrograms(duplicate_emb, parent_emb)
    if convolved.size > 1:
        if fft_dublicate_detector(convolved):
            return True
    return False


def load_embeddings():
    """
    Загружает эмбеддинги видео из памяти
    """
    data = dict()
    for filename in os.listdir(VIDEO_EMBEDDINGS_DIR):
        data[filename[:-4]] = np.load(VIDEO_EMBEDDINGS_DIR + "/" + filename)
    return data


def get_duplicate(sample, audio_sample, skip):
    """
    Основной код программы, сравнивает пару из эмбеддинга видео и аудио со всеми эмбеддингами в data, и возвращает, является ли дубликатом одного из них

    сравнивает видео, при сомнении - аудио.
    """
    sample1 = sample.copy()
    score_pos = similarity_scores(sample1, data)  # нормальное изображение
    if skip != "" and skip in score_pos:
        score_pos[skip] = np.inf
    score_neg = similarity_scores(-sample1, data)  # негатив
    sample1[[0, 1]] = sample1[[1, 0]]
    sample1[[2, 3]] = sample1[[3, 2]]
    score_mirr = similarity_scores(sample1, data)  # отзеркаленое
    if skip != "" and skip in score_mirr:
        score_mirr[skip] = np.inf
    res = ("", np.inf)
    sound_check = []
    for key in score_pos:
        score = min(score_pos[key], score_neg[key], score_mirr[key])
        if score < VID_TRESHOLD and res[1] > score:
            res = (key, score)
        elif score < VID_SOUND_TRESHOLD:
            sound_check.append((key, score))
    if res[0] != "":
        return res[0]
    if len(sound_check) == 0:
        return ""
    for key, _ in sound_check:
        if similar_sound(audio_sample, sound_data[key]):
            return key
    return ""


#######################################################################################################
@app.post("/check-video-duplicate")
async def check_video_duplicate(video_link: VideoLinkRequest):
    # Проверка корректности ссылки
    if not video_link.name.startswith("https"):
        raise HTTPException(
            status_code=400, detail="Некорректная ссылка на видео"
        )

    # Извлечение UUID из ссылки
    uuid = video_link.name.split("/")[-1].replace(".mp4", "")

    # Названия файлов эмбеддингов
    audio_file = f"audio_emb_{uuid}.npy"
    video_file = f"{uuid}.npy"

    # Скачивание видео по ссылке
    video_path = f"/tmp/{uuid}.mp4"  # Временный путь для сохранения видео
    try:
        response = requests.get(video_link.name)
        with open(video_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Не удалось скачать видео")

    # Загрузка видео и создание фреймов
    frames = load(video_path)

    # Создание видео-эмбеддинга
    # video_embedding_path = os.path.join(VIDEO_EMBEDDINGS_DIR, video_file)
    a = get_embed(frames)

    # Извлечение аудио из видео
    audio_data = extract_audio_from_video(video_path)

    # Создание аудио-эмбеддинга (спектрограмма)
    # audio_embedding_path = os.path.join(AUDIO_EMBEDDINGS_DIR, audio_file)
    spectrogram = get_spectrogram(audio_data)

    # Передача эмбеддингов в функцию get_duplicate
    duplicate_result = get_duplicate(a, spectrogram, uuid)

    # Принтуем результат функции
    print("Результат проверки дубликата:", duplicate_result)

    is_duplicate = None
    if duplicate_result != "":
        is_duplicate = True
        duplicate_for = duplicate_result
        return {
            "is_duplicate": is_duplicate,
            "duplicate_for": duplicate_for,
        }
    else:
        is_duplicate = False
        return {
            "is_duplicate": is_duplicate,
            "duplicate_for": "",
        }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Ошибка сервера"},
    )


data = load_embeddings()
sound_data = load_sound_embeddings(AUDIO_EMBEDDINGS_DIR)
