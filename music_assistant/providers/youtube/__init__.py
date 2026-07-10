"""
YouTube Provider for Music Assistant (yt-dlp based).

登入 UI 流程（兩步驟，避免阻塞前端）：

  步驟 1｜使用者按「開始登入」
    → action="start_auth"
    → 向 Google 取得 device code
    → 立即回傳 ALERT 顯示 URL + user_code 給使用者
    → device_code 暫存在 hidden 欄位 "pending_device_code"

  步驟 2｜使用者完成瀏覽器授權後，按「確認已完成授權」
    → action="check_auth"
    → 讀取 pending_device_code 並向 Google 輪詢（最多等 30 秒）
    → 成功後把 token 寫入 values，MA 持久化

對齊 MA 官方 ytmusic provider 設計：
  - SECURE_STRING 儲存 auth_token / refresh_token
  - ALERT 顯示指引訊息
  - ACTION 觸發動作
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import TYPE_CHECKING, AsyncGenerator, Any

# YTM 將英文頻道轉成中文在地名時，若頻道名像西方合輯／劇集／原聲帶，保留英文以利搜尋（如 Glee Cast）。
_YTM_KEEP_ENGLISH_ARTIST_RE = re.compile(
    r"(?i)\b("
    r"glee|\bcast\b|orchestra|symphony|\bband\b|ensemble|choir|soundtrack|"
    r"vevo|official|records|\bost\b|"
    r"various\s+artists|karaoke|tribute|"
    r"original\s+soundtrack|motion\s+picture|cover\s+version"
    r")\b"
)

from music_assistant_models.background_task import TaskSchedule
from music_assistant_models.enums import (
    ConfigEntryType,
    ProviderFeature,
    StreamType,
    ContentType,
    ImageType,
    MediaType,
    TaskScheduleType,
)
from music_assistant_models.media_items import (
    Track,
    Artist,
    Album,
    Playlist,
    MediaItemImage,
    ItemMapping,
    ProviderMapping,
    RecommendationFolder,
    SearchResults,
    AudioFormat,
    UniqueList,
)

try:
    from music_assistant_models.streamdetails import StreamDetails
except ImportError:
    from music_assistant_models.media_items import StreamDetails

from music_assistant_models.errors import (
    MediaNotFoundError,
    UnplayableMediaError,
)

from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.models.server import MusicAssistant
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig

# ---------------------------------------------------------------------------
# 設定欄位 key
# ---------------------------------------------------------------------------
CONF_ACTION_START_AUTH  = "start_auth"        # 步驟 1：取得 device code
CONF_ACTION_CHECK_AUTH  = "check_auth"        # 步驟 2：確認授權完成
CONF_AUTH_TOKEN         = "auth_token"        # access token（SECURE_STRING）
CONF_REFRESH_TOKEN      = "refresh_token"     # refresh token（SECURE_STRING）
CONF_TOKEN_TYPE         = "token_type"        # 通常是 "Bearer"
CONF_EXPIRY_TIME        = "expiry_time"       # expires_in 秒數（字串）
CONF_PENDING_DEVICE_CODE = "pending_device_code"  # 暫存 device_code（hidden）
CONF_COOKIES_FILE       = "cookies_file"      # 備選：手動 cookies
CONF_BGUTIL_URL         = "bgutil_url"        # bgutil PO token server URL

# ---------------------------------------------------------------------------
# OAuth2 端點
# ---------------------------------------------------------------------------
OAUTH2_DEVICE_URL = "https://oauth2.googleapis.com/device/code"
OAUTH2_TOKEN_URL  = "https://oauth2.googleapis.com/token"
OAUTH2_SCOPE      = (
    "https://www.googleapis.com/auth/youtube.readonly "
    "https://www.googleapis.com/auth/youtube"
)

# YouTube Data API v3
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# YouTube InnerTube API（內部 API，無配額限制）
INNERTUBE_BASE    = "https://www.youtube.com/youtubei/v1"
INNERTUBE_KEY     = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"  # 公開 key，所有 YouTube app 共用
INNERTUBE_CONTEXT = {
    "client": {
        "clientName": "TVHTML5",
        "clientVersion": "7.20260114.12.00",
        "hl": "zh-TW",
        "gl": "TW",
    }
}

# YouTube TV OAuth2 Client（公開憑證，Smarttube/NewPipe 等使用）
# 用此 client_id 授權的 token 可存取 InnerTube，且不受 Data API v3 配額限制
YOUTUBE_TV_CLIENT_ID     = "861556708454-d6dlm3lh05idd8npek18k6be8ba3oc68.apps.googleusercontent.com"
YOUTUBE_TV_CLIENT_SECRET = "SboVhoG9s0rNafixCSGGKXAT"

# 設定 key：使用者自填的憑證
CONF_CLIENT_ID     = "client_id"
CONF_CLIENT_SECRET = "client_secret"
CONF_AUTH_MODE     = "auth_mode"  # "custom"（自建）或 "tv"（YouTube TV 內建）
CONF_RECOMMEND_INTERVAL = "recommend_sync_interval_hours"  # 推薦快取刷新間隔
CONF_YTMUSIC_LANGUAGE = "ytmusic_language"  # ytmusicapi YTMusic(language=…)
CONF_YTMUSIC_ENRICH_ARTIST = "ytmusic_enrich_artist"  # get_song 補藝人名：off / title_cjk / always


# ---------------------------------------------------------------------------
# 模組層級函數（MA 框架呼叫）
# ---------------------------------------------------------------------------

async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> MusicProvider:
    # 在 setup 時就傳入靜態 features，確保 MA 框架初始化時就能看到
    # 動態 features（如 LIBRARY_PLAYLISTS）會在 handle_async_init 後更新
    static_features = {
        ProviderFeature.SEARCH,
        ProviderFeature.BROWSE,
        ProviderFeature.ARTIST_TOPTRACKS,
        ProviderFeature.ARTIST_ALBUMS,
    }
    return YouTubeProvider(mass, manifest, config, supported_features=static_features)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, Any] | None = None,
) -> tuple[ConfigEntry, ...]:
    """
    回傳設定欄位。

    MA 的機制：每次 UI 觸發 action 時，都會以目前的 values + action 重新呼叫此函數，
    函數回傳的內容即為 UI 最新狀態，MA 也會把更新後的 values 持久化。

    兩步驟登入流程：
      start_auth → 取得 device code，回傳 ALERT 顯示給使用者
      check_auth → 輪詢 token，寫入 values
    """
    import aiohttp
    from music_assistant_models.config_entries import ConfigEntry

    if values is None:
        values = {}

    # ---------------------------------------------------------------
    # 步驟 1：取得 device code，立即回傳 ALERT（不阻塞）
    # ---------------------------------------------------------------
    if action == CONF_ACTION_START_AUTH:
        auth_mode = values.get(CONF_AUTH_MODE, "tv")
        if auth_mode == "tv":
            client_id = YOUTUBE_TV_CLIENT_ID
        else:
            client_id = values.get(CONF_CLIENT_ID, "").strip()
        if not client_id:
            values["_auth_step"]  = "error"
            values["_auth_error"] = "請先填入 OAuth Client ID 再開始登入。"
        else:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        OAUTH2_DEVICE_URL,
                        data={"client_id": client_id, "scope": OAUTH2_SCOPE},
                    ) as resp:
                        resp_text = await resp.text()
                        if resp.status != 200:
                            # 嘗試解析 Google 回傳的 JSON 錯誤詳情
                            try:
                                err_json = json.loads(resp_text)
                                google_error = err_json.get("error", "")
                                google_desc  = err_json.get("error_description", "")
                                if google_error == "invalid_client":
                                    hint = "\n\n→ OAuth Client ID 無效或未開啟 device flow 授權。TV 模式的公開憑證可能已被 Google 封鎖，請改用「自建 Google Cloud」模式並填入自己的 Client ID/Secret。"
                                elif google_error == "access_denied":
                                    hint = "\n\n→ 授權被拒，可能是 scope 不被此 client 支援。"
                                elif google_error == "org_internal":
                                    hint = "\n\n→ 此 OAuth client 僅限內部組織使用，無法用於個人帳號。"
                                else:
                                    hint = ""
                                raise Exception(
                                    f"HTTP {resp.status} {google_error}: {google_desc}{hint}"
                                )
                            except json.JSONDecodeError:
                                raise Exception(f"HTTP {resp.status}: {resp_text[:300]}")
                        device_data = json.loads(resp_text)

                # 把 device_code 暫存在 values，讓 MA 傳給下一次呼叫
                values[CONF_PENDING_DEVICE_CODE] = json.dumps({
                    "device_code": device_data["device_code"],
                    "interval":    device_data.get("interval", 5),
                    "expires_at":  time.time() + device_data.get("expires_in", 1800),
                })
                values["_auth_step"] = "pending"
                values["_verification_url"] = device_data["verification_url"]
                values["_user_code"]        = device_data["user_code"]

            except Exception as exc:
                values["_auth_step"]  = "error"
                values["_auth_error"] = str(exc)

    # ---------------------------------------------------------------
    # 步驟 2：輪詢 token（使用者已在瀏覽器完成授權）
    # ---------------------------------------------------------------
    elif action == CONF_ACTION_CHECK_AUTH:
        pending_raw = values.get(CONF_PENDING_DEVICE_CODE, "")
        if not pending_raw:
            values["_auth_step"]  = "error"
            values["_auth_error"] = "找不到暫存的 device code，請重新開始登入。"
        else:
            try:
                pending = json.loads(pending_raw)
                device_code = pending["device_code"]
                interval    = pending.get("interval", 5)
                expires_at  = pending.get("expires_at", 0)

                auth_mode = values.get(CONF_AUTH_MODE, "tv")
                if auth_mode == "tv":
                    cid = YOUTUBE_TV_CLIENT_ID
                    csecret = YOUTUBE_TV_CLIENT_SECRET
                else:
                    cid = values.get(CONF_CLIENT_ID, "").strip()
                    csecret = values.get(CONF_CLIENT_SECRET, "").strip()

                poll_payload = {
                    "client_id":     cid,
                    "client_secret": csecret,
                    "device_code":   device_code,
                    "grant_type":    "urn:ietf:params:oauth:grant-type:device_code",
                }

                token_data = None
                # 最多等 30 秒（使用者按下按鈕表示他已在瀏覽器操作完畢）
                deadline = min(time.monotonic() + 30, expires_at - time.time() + time.monotonic())

                async with aiohttp.ClientSession() as session:
                    while time.monotonic() < deadline:
                        async with session.post(OAUTH2_TOKEN_URL, data=poll_payload) as resp:
                            result = await resp.json()

                        error = result.get("error")
                        if error == "authorization_pending":
                            await asyncio.sleep(interval)
                            continue
                        elif error == "slow_down":
                            interval += 5
                            await asyncio.sleep(interval)
                            continue
                        elif error:
                            values["_auth_step"]  = "error"
                            values["_auth_error"] = f"Google 回傳錯誤：{error}"
                            break

                        # 成功
                        token_data = result
                        break

                if token_data and "access_token" in token_data:
                    values[CONF_AUTH_TOKEN]    = token_data["access_token"]
                    values[CONF_REFRESH_TOKEN] = token_data.get("refresh_token", "")
                    values[CONF_TOKEN_TYPE]    = token_data.get("token_type", "Bearer")
                    values[CONF_EXPIRY_TIME]   = str(time.time() + token_data.get("expires_in", 3600))
                    # 清除暫存資料
                    values.pop(CONF_PENDING_DEVICE_CODE, None)
                    values.pop("_auth_step", None)
                    values.pop("_verification_url", None)
                    values.pop("_user_code", None)
                    values.pop("_auth_error", None)
                elif "_auth_step" not in values:
                    # 沒有 error 也沒有 token → 仍在等待
                    values["_auth_step"] = "still_pending"

            except Exception as exc:
                values["_auth_step"]  = "error"
                values["_auth_error"] = str(exc)

    # ---------------------------------------------------------------
    # 組裝 UI ConfigEntry 列表
    # ---------------------------------------------------------------
    entries: list[ConfigEntry] = []

    # --- Instance 名稱（區分多帳號）---
    entries.append(ConfigEntry(
        key="instance_name_postfix",
        type=ConfigEntryType.STRING,
        label="帳號名稱（選填，用於區分多個帳號）",
        description="例如「個人」、「家庭」，會顯示成「YouTube [個人]」。",
        default_value="",
        required=False,
        value=values.get("instance_name_postfix", ""),
    ))

    auth_step   = values.get("_auth_step", "")
    has_token   = bool(values.get(CONF_AUTH_TOKEN))

    # --- 目前狀態提示 ---
    if has_token:
        # 已登入
        entries.append(ConfigEntry(
            key="auth_status",
            type=ConfigEntryType.ALERT,
            label="✅ 已成功登入 YouTube 帳號，串流將使用已認證的身份。",
            required=False,
        ))
    elif auth_step == "pending":
        # 等待使用者完成授權
        verification_url = values.get("_verification_url", "https://www.google.com/device")
        code = values.get("_user_code", "")

        # 步驟說明（純文字 ALERT）
        entries.append(ConfigEntry(
            key="auth_instructions",
            type=ConfigEntryType.ALERT,
            label="請依照以下步驟完成 YouTube 授權：",
            required=False,
        ))

        # 步驟一：URL 用 STRING 唯讀欄位 → 點選即可框選複製，description 說明用途
        entries.append(ConfigEntry(
            key="auth_verification_url",
            type=ConfigEntryType.STRING,
            label="① 複製以下網址，在瀏覽器中開啟",
            description="請複製此網址並在任意瀏覽器（手機或電腦均可）開啟，然後登入您的 Google 帳號完成授權。",
            default_value=verification_url,
            value=verification_url,
            required=False,
        ))

        # 步驟二：代碼用唯讀 STRING → 輸入框天生支援框選複製
        entries.append(ConfigEntry(
            key="auth_user_code",
            type=ConfigEntryType.STRING,
            label="② 在授權頁面輸入以下代碼",
            description="在 Google 授權頁面輸入此代碼以完成驗證。",
            default_value=code,
            value=code,
            required=False,
        ))

        # 步驟三說明
        entries.append(ConfigEntry(
            key="auth_step3",
            type=ConfigEntryType.LABEL,
            label="③ 完成後按下方「確認已完成授權」按鈕",
            required=False,
        ))
    elif auth_step == "still_pending":
        verification_url = values.get("_verification_url", "https://www.google.com/device")
        code = values.get("_user_code", "")
        entries.append(ConfigEntry(
            key="auth_status",
            type=ConfigEntryType.ALERT,
            label="⏳ 尚未偵測到授權完成，請確認已在瀏覽器完成登入後再試一次。",
            required=False,
        ))
        if code:
            entries.append(ConfigEntry(
                key="auth_verification_url",
                type=ConfigEntryType.STRING,
                label="授權頁面網址（可框選複製）",
                default_value=verification_url,
                value=verification_url,
                required=False,
            ))
            entries.append(ConfigEntry(
                key="auth_user_code",
                type=ConfigEntryType.STRING,
                label="授權代碼（可框選複製）",
                default_value=code,
                value=code,
                required=False,
            ))
    elif auth_step == "error":
        error_msg = values.get("_auth_error", "未知錯誤")
        entries.append(ConfigEntry(
            key="auth_status",
            type=ConfigEntryType.ALERT,
            label=f"❌ 登入失敗：{error_msg}\n\n請重新按「開始登入」。",
            required=False,
        ))
    else:
        # 預設：尚未登入，顯示說明
        entries.append(ConfigEntry(
            key="auth_status",
            type=ConfigEntryType.ALERT,
            label=(
                "尚未登入 YouTube 帳號。\n"
                "登入後 yt-dlp 將使用已認證身份，可大幅提升串流穩定性、"
                "避免 YouTube 限速，並允許存取需登入的內容。"
            ),
            required=False,
        ))

    # --- 步驟 1 按鈕：開始登入 ---
    if not has_token:
        entries.append(ConfigEntry(
            key=CONF_ACTION_START_AUTH,
            type=ConfigEntryType.ACTION,
            label="🔐 開始登入（取得授權代碼）",
            description="點擊後頁面將顯示授權 URL 和代碼，請在瀏覽器中完成操作。",
            action=CONF_ACTION_START_AUTH,
            action_label="開始登入",
            required=False,
        ))

    # --- 步驟 2 按鈕：確認授權（只在等待時顯示）---
    if auth_step in ("pending", "still_pending"):
        entries.append(ConfigEntry(
            key=CONF_ACTION_CHECK_AUTH,
            type=ConfigEntryType.ACTION,
            label="✅ 我已完成授權（點擊確認並取得 Token）",
            description="在瀏覽器完成 Google 帳號授權後，點此按鈕完成登入。",
            action=CONF_ACTION_CHECK_AUTH,
            action_label="確認已完成授權",
            required=False,
        ))

    # --- 登出按鈕（已登入時顯示）---
    if has_token:
        entries.append(ConfigEntry(
            key="logout",
            type=ConfigEntryType.ACTION,
            label="🚪 登出 YouTube 帳號",
            action="logout",
            action_label="登出",
            required=False,
        ))

    # --- API 模式說明與選擇 ---
    entries.append(ConfigEntry(key="divider_api", type=ConfigEntryType.DIVIDER, label=""))
    entries.append(ConfigEntry(
        key="api_mode_title",
        type=ConfigEntryType.LABEL,
        label="⚙️ API 模式",
        required=False,
    ))
    entries.append(ConfigEntry(
        key="api_mode_info",
        type=ConfigEntryType.ALERT,
        label=(
            "【模式比較】\n\n"
            "🔴 僅 yt-dlp\n"
            "  速度慢（1-3s）| 零配額 | 設定最簡單\n"
            "  Fallback：yt-dlp only\n\n"
            "🟡 自建 Google Cloud（Data API v3）\n"
            "  速度快（100-400ms）| 10,000 units/天 | 需建 Cloud 專案\n"
            "  Fallback：Data API v3 → InnerTube → yt-dlp\n\n"
            "🟢 YouTube TV（InnerTube）建議\n"
            "  速度快（200-400ms）| 零配額 | 無需 Cloud 專案\n"
            "  Fallback：InnerTube → yt-dlp\n\n"
            "【功能支援】✅可用  ⚠️需條件  ❌不可用\n\n"
            "功能               yt-dlp  custom   tv\n"
            "搜尋影片/清單/頻道   ✅      ✅       ✅\n"
            "播放影片串流         ✅      ✅       ✅\n"
            "瀏覽頻道影片         ✅      ✅       ✅\n"
            "公開播放清單         ✅      ✅       ✅\n"
            "個人播放清單列表   ⚠️cookies ⚠️OAuth  ⚠️OAuth\n"
            "個人清單(PLAOYGtd_)  ❌    ⚠️OAuth  ⚠️OAuth\n"
            "年齡限制影片       ⚠️cookies ✅OAuth  ✅OAuth\n"
            "私人影片           ⚠️cookies ✅OAuth  ✅OAuth\n\n"
            "⚠️ cookies = 需設定 Cookies 檔案路徑\n"
            "✅ OAuth   = 已完成 OAuth 登入即可，不需 cookies\n"
            "⚠️ OAuth   = 需完成 OAuth 登入"
        ),
        required=False,
    ))

    auth_mode = values.get(CONF_AUTH_MODE, "tv")
    from music_assistant_models.config_entries import ConfigValueOption
    entries.append(ConfigEntry(
        key=CONF_AUTH_MODE,
        type=ConfigEntryType.STRING,
        label="API 模式選擇",
        default_value="tv",
        required=False,
        value=auth_mode,
        options=[
            ConfigValueOption(title="🟢 YouTube TV（InnerTube，建議）", value="tv"),
            ConfigValueOption(title="🟡 自建 Google Cloud（Data API v3）", value="custom"),
            ConfigValueOption(title="🔴 僅 yt-dlp（無需 API）", value="ytdlp"),
        ],
        multi_value=False,
    ))

    # --- 模式說明 / 自建憑證（依模式顯示）---
    if auth_mode == "tv":
        entries.append(ConfigEntry(
            key="tv_mode_note",
            type=ConfigEntryType.ALERT,
            label=(
                "✅ 已選擇 YouTube TV 模式。\n"
                "使用內建公開憑證（與 Smarttube、NewPipe 相同），無需自建 Google Cloud 專案。\n"
                "點擊「開始登入」即可授權，授權頁面會顯示為「在電視上使用 YouTube」。\n\n"
                "⚠️ 注意：若出現 403 invalid_client 錯誤，表示 Google 已封鎖此公開憑證（伺服器 IP 尤其常見）。\n"
                "解決方式：切換到「自建 Google Cloud」模式，使用自己建立的 OAuth 憑證。\n"
                "建立步驟：Google Cloud Console → APIs & Services → Credentials → "
                "Create OAuth client ID（類型選「TV and Limited Input devices」）→ 啟用 YouTube Data API v3"
            ),
            required=False,
        ))
    elif auth_mode == "custom":
        entries.append(ConfigEntry(key="divider_credentials", type=ConfigEntryType.DIVIDER, label=""))
        entries.append(ConfigEntry(
            key="credentials_title",
            type=ConfigEntryType.LABEL,
            label="🔑 Google Cloud OAuth 憑證",
            required=False,
        ))
        entries.append(ConfigEntry(
            key="custom_mode_note",
            type=ConfigEntryType.ALERT,
            label=(
                "需要在 Google Cloud Console 建立專案並啟用 YouTube Data API v3。\n"
                "建立方式：APIs & Services → Credentials → Create OAuth client ID\n"
                "（應用程式類型選「TV and Limited Input devices」）\n"
                "Fallback 順序：Data API v3 → InnerTube → yt-dlp\n"
                "配額用完（403）後自動切換 InnerTube，不需要重啟。"
            ),
            required=False,
        ))
        entries.append(ConfigEntry(
            key=CONF_CLIENT_ID,
            type=ConfigEntryType.STRING,
            label="OAuth Client ID",
            description="Google Cloud Console 取得的 OAuth 2.0 Client ID。",
            default_value="",
            required=False,
            value=values.get(CONF_CLIENT_ID, ""),
        ))
        entries.append(ConfigEntry(
            key=CONF_CLIENT_SECRET,
            type=ConfigEntryType.SECURE_STRING,
            label="OAuth Client Secret",
            description="對應 Client ID 的 Client Secret。",
            default_value="",
            required=False,
            value=values.get(CONF_CLIENT_SECRET, ""),
        ))
    else:  # ytdlp
        entries.append(ConfigEntry(
            key="ytdlp_mode_note",
            type=ConfigEntryType.ALERT,
            label=(
                "ℹ️ 純 yt-dlp 模式：不使用任何 API，也不需要登入。\n"
                "個人播放清單功能不可用（除非提供 Cookies 檔案）。\n"
                "搜尋和播放速度較慢，但完全不依賴 Google API。"
            ),
            required=False,
        ))

    entries.append(ConfigEntry(key="divider_1", type=ConfigEntryType.DIVIDER, label=""))

    # --- Hidden 欄位（token 儲存）---
    entries += [
        ConfigEntry(
            key=CONF_AUTH_TOKEN,
            type=ConfigEntryType.SECURE_STRING,
            label="Access Token",
            default_value="",
            required=False,
            hidden=True,
            value=values.get(CONF_AUTH_TOKEN, ""),
        ),
        ConfigEntry(
            key=CONF_REFRESH_TOKEN,
            type=ConfigEntryType.SECURE_STRING,
            label="Refresh Token",
            default_value="",
            required=False,
            hidden=True,
            value=values.get(CONF_REFRESH_TOKEN, ""),
        ),
        ConfigEntry(
            key=CONF_TOKEN_TYPE,
            type=ConfigEntryType.STRING,
            label="Token Type",
            default_value="Bearer",
            required=False,
            hidden=True,
            value=values.get(CONF_TOKEN_TYPE, "Bearer"),
        ),
        ConfigEntry(
            key=CONF_EXPIRY_TIME,
            type=ConfigEntryType.STRING,
            label="Token Expiry",
            default_value="",
            required=False,
            hidden=True,
            value=values.get(CONF_EXPIRY_TIME, ""),
        ),
        ConfigEntry(
            key=CONF_PENDING_DEVICE_CODE,
            type=ConfigEntryType.STRING,
            label="Pending Device Code",
            default_value="",
            required=False,
            hidden=True,
            value=values.get(CONF_PENDING_DEVICE_CODE, ""),
        ),
    ]

    # --- 處理登出動作（清除 token）---
    if action == "logout":
        for entry in entries:
            if entry.key in (CONF_AUTH_TOKEN, CONF_REFRESH_TOKEN,
                             CONF_TOKEN_TYPE, CONF_EXPIRY_TIME,
                             CONF_PENDING_DEVICE_CODE):
                entry.value = ""

    # --- 備選：手動 cookies ---
    entries.append(ConfigEntry(key="divider_2", type=ConfigEntryType.DIVIDER, label=""))
    entries.append(ConfigEntry(
        key=CONF_COOKIES_FILE,
        type=ConfigEntryType.STRING,
        label="Cookies 檔案路徑（選填）",
        description=(
            "Netscape 格式的 cookies.txt 完整路徑，例如 /config/yt_cookies.txt。\n\n"
            "tv / custom 模式下：年齡限制和私人影片透過 InnerTube + OAuth 直接存取，"
            "通常不需要 cookies。\n"
            "yt-dlp 模式下：年齡限制和私人影片需要此 cookies 檔案。\n\n"
            "匯出方式：在 Chrome/Firefox 使用 Get cookies.txt 擴充功能匯出 YouTube cookies。"
        ),
        default_value="",
        required=False,
        value=values.get(CONF_COOKIES_FILE, ""),
    ))

    entries.append(ConfigEntry(key="divider_recommend", type=ConfigEntryType.DIVIDER, label=""))
    entries.append(ConfigEntry(
        key=CONF_RECOMMEND_INTERVAL,
        type=ConfigEntryType.INTEGER,
        label="推薦自動刷新間隔（小時，0 = 不自動刷新）",
        description=(
            "背景定期向 YouTube Music 取得個人化推薦並快取，"
            "開啟推薦頁時直接從快取讀取，不再即時等待。\n"
            "需要登入帳號（OAuth 或 Cookies）才有個人化推薦。"
        ),
        default_value=3,
        required=False,
        value=values.get(CONF_RECOMMEND_INTERVAL, 3),
    ))

    entries.append(ConfigEntry(
        key=CONF_YTMUSIC_LANGUAGE,
        type=ConfigEntryType.STRING,
        label="YouTube Music API 語系（ytmusicapi language）",
        description=(
            "傳給 ytmusicapi.YTMusic(language=…)，影響 YTMusic 搜尋／藝人／推薦等回傳語言。\n"
            "常見值：en、zh_TW、zh_CN、ja、ko（依 ytmusicapi 支援為準）。\n"
            "留空則使用 en。"
        ),
        default_value="",
        required=False,
        value=values.get(CONF_YTMUSIC_LANGUAGE, ""),
    ))

    entries.append(ConfigEntry(
        key=CONF_YTMUSIC_ENRICH_ARTIST,
        type=ConfigEntryType.STRING,
        label="YTM 補齊藝人名（get_song）",
        description=(
            "off：不呼叫 get_song 覆寫藝人，搜尋用字與 Data API 一致（全英文）。\n"
            "title_cjk：僅當影片標題含中日韓字時才用 YTM 作者名（省 API，英文歌不本地化）。\n"
            "always：盡量採用 YTM author；但若頻道名像 Glee／soundtrack 等西方合輯，仍保留英文。"
        ),
        default_value="always",
        required=False,
        value=values.get(CONF_YTMUSIC_ENRICH_ARTIST, "always"),
    ))

    return tuple(entries)


# ---------------------------------------------------------------------------
# Provider 主體
# ---------------------------------------------------------------------------

class YouTubeProvider(MusicProvider):
    """YouTube yt-dlp 音樂 Provider，支援 OAuth2 Device Flow 登入。"""

    _client_id: str = ""
    _client_secret: str = ""
    _access_token: str = ""
    _refresh_token: str = ""
    _token_type: str = "Bearer"
    _token_expires_at: float = 0.0
    _cookies_file: str = ""
    _bgutil_url: str = ""                       # bgutil PO token server（若設定）
    _auth_mode: str = "tv"             # "tv" | "custom" | "ytdlp"
    _api_quota_exceeded: bool = False  # Data API v3 配額用完時設為 True，改用 yt-dlp
    _api_quota_reset_at: float = 0.0   # 隔天 UTC 00:00 重置
    _http_session: Any = None          # 共用 aiohttp session（避免每次請求重建）
    _ytdlp_clients_synced: bool = False  # 是否已從 yt-dlp 同步過客戶端版本
    _ytmusic: Any = None                # ytmusicapi.YTMusic instance
    _direct_url_cache: dict[str, tuple[float, dict]] = {}
    _DIRECT_URL_CACHE_TTL: float = 5 * 3600  # YouTube URL 有效期約 6h，快取 5h
    _fast_client_fail_until: float = 0.0  # android_vr 失敗後暫時跳過的截止時間
    _FAST_CLIENT_COOLDOWN: float = 600  # android_vr 失敗後冷卻 10 分鐘
    _yt_dlp_module: Any = None          # yt_dlp 模組（懶載入，避免 subprocess）
    _prefetch_in_progress: set[str] = set()  # 正在背景預取中的 item_id
    _prefetch_task: Any = None               # 當前的背景預取 asyncio.Task
    _PREFETCH_AHEAD: int = 30                 # 預取佇列中接下來幾首
    _recommendation_cache: list | None = None  # 快取的推薦結果
    _recommendation_cache_at: float = 0.0      # 快取建立時間
    _ytmusic_language: str = ""                # ytmusicapi 語系（空則 en）
    _ytmusic_enrich_artist: str = "always"     # get_song 補藝人名策略
    _cookies_cache: tuple[float, dict[str, str]] | None = None  # cookies.txt 解析快取（mtime, cookies）
    _token_lock: Any = None                    # 防止並發 token refresh
    _ytm_author_cache: dict[str, tuple[float, str]] = {}  # get_song author 快取（TTL 24h）

    # MA provider key → yt-dlp client key
    _YTDLP_CLIENT_MAP: dict[str, str] = {
        "android_vr": "android_vr",
        "android_embedded": "ios",
        "tv": "tv",
        "android": "android",
        "web": "web",
    }

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    # 用於 subprocess 提取 yt-dlp INNERTUBE_CLIENTS 的腳本
    _YTDLP_EXTRACT_SCRIPT = """
import json
try:
    from yt_dlp.extractor.youtube._base import INNERTUBE_CLIENTS
except ImportError:
    from yt_dlp.extractor.youtube import INNERTUBE_CLIENTS
d = {}
for k, v in INNERTUBE_CLIENTS.items():
    c = v.get('INNERTUBE_CONTEXT', {}).get('client', {})
    d[k] = {'ver': c.get('clientVersion', ''), 'ua': c.get('userAgent', '')}
print(json.dumps(d))
"""

    @classmethod
    def _sync_innertube_clients_from_ytdlp(cls, logger: Any = None) -> None:
        """從 yt-dlp 的 INNERTUBE_CLIENTS 動態同步客戶端版本號。

        MA 跑在 venv 裡，yt-dlp 裝在系統 Python，無法直接 import。
        透過 subprocess 呼叫 yt-dlp 的 Python 直譯器提取資料。
        """
        if cls._ytdlp_clients_synced:
            return
        cls._ytdlp_clients_synced = True

        import subprocess, shutil

        # 從 yt-dlp 的 shebang 找到它的 Python 直譯器
        python_bin = None
        ytdlp_bin = shutil.which("yt-dlp")
        if ytdlp_bin:
            try:
                with open(ytdlp_bin) as f:
                    line = f.readline()
                if line.startswith("#!"):
                    candidate = line[2:].strip().split()[0]
                    if os.path.isfile(candidate):
                        python_bin = candidate
            except Exception:
                pass

        if not python_bin:
            if logger:
                logger.warning(
                    "[YouTube] 找不到 yt-dlp 的 Python 直譯器，"
                    "使用硬編碼版本"
                )
            return

        try:
            result = subprocess.run(
                [python_bin, "-c", cls._YTDLP_EXTRACT_SCRIPT],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                if logger:
                    logger.warning(
                        "[YouTube] yt-dlp INNERTUBE_CLIENTS 提取失敗: %s",
                        result.stderr.strip()[:200],
                    )
                return
            yt_data: dict = json.loads(result.stdout.strip())
        except Exception as exc:
            if logger:
                logger.warning(
                    "[YouTube] 無法從 yt-dlp 讀取 INNERTUBE_CLIENTS: %s",
                    exc,
                )
            return

        updated: list[str] = []
        for ma_key, ytdlp_key in cls._YTDLP_CLIENT_MAP.items():
            if ma_key not in cls._INNERTUBE_CLIENTS:
                continue
            yt_info = yt_data.get(ytdlp_key)
            if not yt_info:
                continue

            new_ver = yt_info.get("ver", "")
            new_ua = yt_info.get("ua", "")
            if not new_ver:
                continue

            ma_cfg = cls._INNERTUBE_CLIENTS[ma_key]
            old_ver = ma_cfg["context"]["client"].get("clientVersion", "")
            if new_ver == old_ver:
                continue

            # --- 更新 context.client ---
            ma_cfg["context"]["client"]["clientVersion"] = new_ver
            if new_ua:
                ma_cfg["context"]["client"]["userAgent"] = new_ua
            elif "userAgent" in ma_cfg["context"]["client"] and old_ver:
                ma_cfg["context"]["client"]["userAgent"] = (
                    ma_cfg["context"]["client"]["userAgent"]
                    .replace(old_ver, new_ver)
                )

            # --- 更新 headers ---
            hdrs = ma_cfg.get("headers", {})
            if "X-YouTube-Client-Version" in hdrs:
                hdrs["X-YouTube-Client-Version"] = new_ver
            if new_ua and "User-Agent" in hdrs:
                hdrs["User-Agent"] = new_ua
            elif "User-Agent" in hdrs and old_ver:
                hdrs["User-Agent"] = hdrs["User-Agent"].replace(
                    old_ver, new_ver
                )

            updated.append(f"{ma_key}: {old_ver} → {new_ver}")

        if updated and logger:
            logger.info(
                "[YouTube] 已從 yt-dlp 同步 InnerTube 客戶端版本: %s",
                ", ".join(updated),
            )
        elif logger:
            logger.debug(
                "[YouTube] yt-dlp InnerTube 客戶端版本與硬編碼版本一致"
            )

    async def handle_async_init(self) -> None:
        """MA 建立 provider instance 後呼叫，用於初始化認證資訊。"""
        self._direct_url_cache = {}
        self._fast_client_fail_until = 0.0
        self._innertube_fail_until = 0.0
        self._prefetch_in_progress = set()
        self._prefetch_task = None
        self._cookies_cache = None
        self._ytm_author_cache = {}
        self._token_lock = asyncio.Lock()
        # yt-dlp 版本同步走 subprocess（最長 30s），放到 executor 避免阻塞事件迴圈
        await asyncio.get_running_loop().run_in_executor(
            None, self._sync_innertube_clients_from_ytdlp, self.logger
        )
        self._client_id     = self.config.get_value(CONF_CLIENT_ID) or ""
        self._client_secret = self.config.get_value(CONF_CLIENT_SECRET) or ""
        self._auth_mode = self.config.get_value(CONF_AUTH_MODE) or "tv"
        self._access_token  = self.config.get_value(CONF_AUTH_TOKEN) or ""
        self._refresh_token = self.config.get_value(CONF_REFRESH_TOKEN) or ""
        self._token_type    = self.config.get_value(CONF_TOKEN_TYPE) or "Bearer"

        expiry_raw = self.config.get_value(CONF_EXPIRY_TIME) or ""
        if expiry_raw:
            try:
                # CONF_EXPIRY_TIME 存的是絕對時間戳（time.time() + expires_in）
                # 舊版存的是相對秒數，做相容處理：若值合理大才當作絕對時間
                val = float(expiry_raw)
                if val > 1_000_000_000:  # 合理的 Unix timestamp（> 2001年）
                    self._token_expires_at = val
                else:
                    # 舊格式：相對秒數，換算成絕對時間
                    self._token_expires_at = time.time() + val
            except ValueError:
                self._token_expires_at = 0.0

        cookies_path = self.config.get_value(CONF_COOKIES_FILE) or ""
        if cookies_path and os.path.isfile(cookies_path):
            self._cookies_file = cookies_path

        self._bgutil_url = (self.config.get_value(CONF_BGUTIL_URL) or "").strip().rstrip("/")

        self._ytmusic_language = (self.config.get_value(CONF_YTMUSIC_LANGUAGE) or "").strip()
        self._ytmusic_enrich_artist = (
            (self.config.get_value(CONF_YTMUSIC_ENRICH_ARTIST) or "always").strip().lower()
        )

        self.logger.info(
            f"[YouTube] instance_id={self.instance_id!r}, domain={self.domain!r}, "
            f"auth_mode={self._auth_mode!r}"
        )
        if self._access_token and self._cookies_file:
            mode_label = {
                "tv": "TV OAuth (InnerTube 串流 + Cookie 個人頁面)",
                "custom": "Data API v3 + Cookie 個人頁面",
            }.get(self._auth_mode, self._auth_mode)
            self.logger.info(f"[YouTube] {mode_label}")
        elif self._access_token:
            mode_label = {
                "tv": "TV OAuth (InnerTube 串流/瀏覽/搜尋)",
                "custom": "Data API v3 (配額耗盡自動切 InnerTube)",
            }.get(self._auth_mode, self._auth_mode)
            self.logger.info(f"[YouTube] {mode_label}")
        elif self._cookies_file:
            self.logger.info(f"[YouTube] Cookies 模式: {self._cookies_file}")
        else:
            self.logger.info("[YouTube] 匿名模式（建議登入帳號以提升穩定性）")

        # 初始化 ytmusicapi（用於 similar_tracks / recommendations）
        self._init_ytmusic()

        # 根據登入狀態更新 MA 的 _supported_features（base class 用此決定功能）
        self._update_supported_features()
        self.logger.info(
            f"[YouTube] supported_features={[f.value for f in self._supported_features]}"
        )

        # 修正舊版本存入的錯誤 provider_instance（domain 而非 instance_id）
        await self._fix_legacy_provider_mappings()

        # 註冊推薦快取定期刷新背景工作
        interval = int(self.config.get_value(CONF_RECOMMEND_INTERVAL) or 0)
        if interval > 0 and self._ytmusic:
            self.mass.tasks.register_scheduled_task(
                task_id=f"{self.instance_id}.sync_recommendations",
                name="YouTube Music 推薦刷新",
                handler=self._sync_recommendations,
                schedule=TaskSchedule(
                    type=TaskScheduleType.HOURLY,
                    every=interval,
                ),
                initial_delay=90.0,  # 等其他初始化完成後再開始
                allow_retry=True,
                allow_cancel=True,
            )

    async def update_config(self, config: ProviderConfig, changed_keys: set[str]) -> None:
        """Handle config update.

        Override base class to work around MA 2.8.x bug where base class calls:
          mass.call_later(1, mass.load_provider_config, config, self.instance_id)
        but load_provider_config only accepts one positional argument (prov_conf).
        We replicate the correct behaviour: schedule reload with only config.
        """
        try:
            from music_assistant.constants import CONF_LOG_LEVEL  # noqa: PLC0415
            if f"values/{CONF_LOG_LEVEL}" in changed_keys:
                if hasattr(self, "_set_log_level_from_config"):
                    self._set_log_level_from_config(config)
        except ImportError:
            pass

        # 只有真正需要 reload 的欄位才觸發（排除 token 自動刷新和臨時狀態欄位）
        _no_reload_keys = {
            f"values/{CONF_AUTH_TOKEN}",
            f"values/{CONF_REFRESH_TOKEN}",
            f"values/{CONF_EXPIRY_TIME}",
            f"values/{CONF_TOKEN_TYPE}",
            f"values/{CONF_PENDING_DEVICE_CODE}",
        }
        value_keys_changed = {
            k for k in changed_keys
            if k.startswith("values/") and k not in _no_reload_keys
        }
        if value_keys_changed:
            self.logger.info(
                "Config updated, reloading provider %s (instance_id=%s)",
                self.domain,
                self.instance_id,
            )
            task_id = f"provider_reload_{self.instance_id}"
            # 正確呼叫：只傳 config，修正 MA 2.8.x base class 多傳 instance_id 的 bug
            self.mass.call_later(
                1, self.mass.load_provider_config, config, task_id=task_id
            )

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        :param is_removed: True if the provider is being removed (not just reloaded).
        """
        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
        self._prefetch_task = None
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

    # ------------------------------------------------------------------
    # 功能宣告
    # ------------------------------------------------------------------

    async def _fix_legacy_provider_mappings(self) -> None:
        """修正舊版本將 provider_instance 存成 domain 的問題。"""
        try:
            db = self.mass.music.database
            # 找出 provider_instance = domain（舊格式）的 mapping，更新成 instance_id
            rows = await db.get_rows(
                "provider_mappings",
                {
                    "provider_domain": self.domain,
                    "provider_instance": self.domain,  # 舊的錯誤格式
                },
            )
            if not rows:
                return
            self.logger.info(
                f"[YouTube] 修正 {len(rows)} 筆舊版 provider_mappings "
                f"（provider_instance: '{self.domain}' → '{self.instance_id}'）"
            )
            await db.execute(
                "UPDATE provider_mappings SET provider_instance = ? "
                "WHERE provider_domain = ? AND provider_instance = ?",
                (self.instance_id, self.domain, self.domain),
            )
        except Exception as exc:
            self.logger.warning(f"[YouTube] 修正舊版 mappings 失敗: {exc}")

    async def _try_methods(
        self,
        operation: str,
        methods: list,
        *args,
        is_generator: bool = False,
        **kwargs,
    ):
        """
        依序嘗試多個方法，全部失敗時拋出包含完整錯誤原因的 exception。

        methods: list of (label, coroutine_func) 或 (label, async_generator_func)
        operation: 操作名稱（用於 log 和錯誤訊息）
        """
        from music_assistant_models.errors import MusicAssistantError
        errors: list[str] = []

        for label, func in methods:
            try:
                if is_generator:
                    results = []
                    async for item in func(*args, **kwargs):
                        results.append(item)
                    self.logger.debug(
                        f"[YouTube] {operation} 成功（{label}），共 {len(results)} 筆"
                    )
                    return results
                else:
                    result = await func(*args, **kwargs)
                    is_last = (label == methods[-1][0])
                    # 搜尋/列表類：空 list 且不是最後一個方法時，繼續嘗試下一個
                    if isinstance(result, list) and len(result) == 0 and not is_last:
                        errors.append(f"{label}: 回傳空結果")
                        self.logger.debug(
                            f"[YouTube] {operation} {label} 回傳空結果，嘗試下一個"
                        )
                        continue
                    self.logger.debug(f"[YouTube] {operation} 成功（{label}），共 {len(result) if isinstance(result, list) else 1} 筆")
                    return result
            except MusicAssistantError:
                raise  # MediaNotFoundError 等不需 fallback，直接往上拋
            except Exception as exc:
                err_str = str(exc)
                if "403" in err_str or "quota" in err_str.lower():
                    self._handle_quota_exceeded()
                    errors.append(f"{label}: 配額超限（已切換模式）")
                else:
                    errors.append(f"{label}: {err_str[:300]}")
                self.logger.warning(
                    f"[YouTube] {operation} {label} 失敗，嘗試下一個: {err_str[:300]}"
                )

        # 全部失敗
        reason = " | ".join(errors)
        self.logger.error(f"[YouTube] {operation} 所有方法均失敗: {reason}")
        raise UnplayableMediaError(
            f"YouTube {operation} 失敗，已嘗試 {len(methods)} 種方式：{reason}"
        )

    def _is_api_available(self) -> bool:
        """檢查是否可用 Data API v3（僅 custom 模式）。
        TV 模式的 OAuth token 用於 InnerTube，不走 Data API v3。
        """
        if self._auth_mode != "custom":
            return False
        if not self._access_token:
            return False
        if self._api_quota_exceeded:
            if time.time() >= self._api_quota_reset_at:
                self._api_quota_exceeded = False
                self.logger.info("[YouTube] Data API v3 配額已重置")
            else:
                return False
        return True

    def _is_innertube_available(self) -> bool:
        """檢查 InnerTube 是否可用。
        TV 和 custom 模式皆可（custom 模式配額耗盡後自動切 InnerTube）。
        ytdlp 模式也可用（匿名 InnerTube，不需 token）。
        """
        return True

    async def _innertube_request(
        self, endpoint: str, payload: dict, use_web_client: bool = False
    ) -> dict:
        """發送 InnerTube API 請求。

        use_web_client=True 時用 WEB client（FEplaylists 等個人頁面需要）；
        否則用 TVHTML5 client（串流、搜尋等）。
        """
        import copy

        url = f"{INNERTUBE_BASE}/{endpoint}?prettyPrint=false"

        if use_web_client:
            web_cfg = self._INNERTUBE_CLIENTS.get("web", {})
            web_ctx = web_cfg.get("context", {}).get("client", {})
            context = {
                "client": {
                    "clientName": web_ctx.get("clientName", "WEB"),
                    "clientVersion": web_ctx.get("clientVersion", "2.20260114.08.00"),
                    "hl": "zh-TW",
                    "gl": "TW",
                }
            }
            client_name_num = web_cfg.get("headers", {}).get(
                "X-YouTube-Client-Name", "1"
            )
            client_version = web_ctx.get("clientVersion", "2.20260114.08.00")
        else:
            tv_cfg = self._INNERTUBE_CLIENTS.get("tv", {})
            tv_ctx = tv_cfg.get("context", {}).get("client", {})
            context = copy.deepcopy(
                tv_cfg.get("context", INNERTUBE_CONTEXT)
            )
            client_name_num = tv_cfg.get("headers", {}).get(
                "X-YouTube-Client-Name", "7"
            )
            client_version = tv_ctx.get(
                "clientVersion", "7.20260114.12.00"
            )

        body    = {"context": context, **payload}
        headers = {
            "Content-Type":             "application/json",
            "Accept-Language":          "zh-TW,zh;q=0.9,en;q=0.8",
            "X-YouTube-Client-Name":    client_name_num,
            "X-YouTube-Client-Version": client_version,
            "Origin":  "https://www.youtube.com",
            "Referer": "https://www.youtube.com/",
        }
        # TV client（非 WEB）：注入 OAuth Bearer token（TV / custom 模式均適用）
        # WEB client：不送 Bearer（會 400），如有 cookies 則走 SAPISIDHASH
        if self._access_token and not use_web_client:
            headers["Authorization"]   = f"Bearer {self._access_token}"
            headers["X-Goog-AuthUser"] = "0"
        elif use_web_client and self._cookies_file:
            headers.update(self._innertube_cookie_headers())
        async with self._session.post(url, json=body, headers=headers) as resp:
            if resp.status == 400:
                body_text = await resp.text()
                raise RuntimeError(f"InnerTube {endpoint} HTTP 400: {body_text[:200]}")
            resp.raise_for_status()
            return await resp.json()

    @property
    def _session(self) -> "aiohttp.ClientSession":
        """取得共用 aiohttp ClientSession（懶建立，自動偵測關閉後重建）。"""
        import aiohttp
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                # 設定 timeout 避免 InnerTube 偶發不回應時請求無限期卡住
                timeout=aiohttp.ClientTimeout(total=30, connect=10),
                # DNS 快取 + 連線重用，降低每個 API 請求的建立開銷
                connector=aiohttp.TCPConnector(limit=20, ttl_dns_cache=300),
            )
        return self._http_session

    def _handle_quota_exceeded(self) -> None:
        """標記 API 配額超限，計算下次重置時間。"""
        import datetime
        self._api_quota_exceeded = True
        # YouTube 配額每天 UTC 00:00 重置
        now = datetime.datetime.now(datetime.timezone.utc)
        tomorrow = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
        self._api_quota_reset_at = tomorrow.timestamp()
        self.logger.warning(
            f"[YouTube] Data API v3 配額用完，切換到 yt-dlp 模式。"
            f"預計重置時間: {tomorrow.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    def _init_ytmusic(self) -> None:
        """初始化 ytmusicapi YTMusic instance（同步呼叫，在 handle_async_init 中使用）。"""
        try:
            import ytmusicapi  # noqa: PLC0415
        except ImportError:
            self.logger.debug(
                "[YouTube] ytmusicapi 不可用，SIMILAR_TRACKS / RECOMMENDATIONS / YTMusic 搜尋停用"
            )
            self._ytmusic = None
            return

        auth: dict | str | None = None
        if self._cookies_file:
            cookie_header = self._cookies_to_ytmusic_header(self._cookies_file)
            if cookie_header:
                auth = cookie_header
                self.logger.debug("[YouTube] ytmusicapi 使用 cookie 認證")

        lang = (getattr(self, "_ytmusic_language", "") or "").strip() or "en"
        try:
            self._ytmusic = ytmusicapi.YTMusic(auth=auth, language=lang)
            self.logger.info(
                f"[YouTube] ytmusicapi {ytmusicapi.__version__} 初始化成功"
                f"（auth={'cookie' if auth else 'anonymous'}, language={lang!r}）"
            )
        except Exception as exc:
            self.logger.warning(f"[YouTube] ytmusicapi 初始化失敗: {exc}")
            self._ytmusic = None

    @staticmethod
    def _cookies_to_ytmusic_header(cookies_file: str) -> str | None:
        """將 Netscape cookies.txt 轉換為 ytmusicapi browser auth 所需的 JSON 字串。

        ytmusicapi 判斷 AuthType.BROWSER 的條件：
        authorization header 必須包含 'SAPISIDHASH'。
        """
        import hashlib  # noqa: PLC0415

        cookies: list[str] = []
        sapisid_value = ""
        try:
            with open(cookies_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 7:
                        domain = parts[0]
                        name = parts[5]
                        value = parts[6]
                        if ".youtube.com" in domain or ".google.com" in domain:
                            cookies.append(f"{name}={value}")
                            if name in ("SAPISID", "__Secure-3PAPISID"):
                                sapisid_value = value
        except Exception:
            return None
        if not cookies or not sapisid_value:
            return None

        origin = "https://music.youtube.com"
        ts = str(int(time.time()))
        sha1 = hashlib.sha1(f"{ts} {sapisid_value} {origin}".encode()).hexdigest()
        sapisidhash = f"SAPISIDHASH {ts}_{sha1}"

        import json as _json  # noqa: PLC0415
        return _json.dumps({
            "authorization": sapisidhash,
            "cookie": "; ".join(cookies),
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/json",
            "X-Goog-AuthUser": "0",
            "x-origin": origin,
        })

    def _update_supported_features(self) -> None:
        """根據目前登入狀態更新 _supported_features。

        個人播放清單：
          tv 模式     → InnerTube TV client + Bearer 取得
          custom 模式 → Data API v3 取得
          ytdlp 模式  → 需 cookies.txt
        """
        features = {
            ProviderFeature.SEARCH,
            ProviderFeature.BROWSE,
            ProviderFeature.ARTIST_TOPTRACKS,
            ProviderFeature.ARTIST_ALBUMS,
        }
        if self._access_token or self._cookies_file:
            features.add(ProviderFeature.LIBRARY_PLAYLISTS)
        if self._ytmusic is not None:
            features.add(ProviderFeature.SIMILAR_TRACKS)
            features.add(ProviderFeature.RECOMMENDATIONS)
        self._supported_features = features

    @property
    def is_streaming_provider(self) -> bool:
        """每個 instance 是不同帳號，資料不同，設為 False 讓 MA 對每個 instance 都查詢。"""
        return False

    @property
    def instance_name_postfix(self) -> str | None:
        """回傳使用者自填的帳號名稱，顯示成「YouTube [名稱]」。"""
        return self.config.get_value("instance_name_postfix") or None

    @property
    def is_logged_in(self) -> bool:
        return bool(self._access_token)

    # ------------------------------------------------------------------
    # Token 管理
    # ------------------------------------------------------------------

    async def _ensure_fresh_token(self) -> None:
        """Token 快過期時自動刷新（提前 5 分鐘），以 lock 防止並發重複刷新。"""
        if not self._access_token:
            return
        if time.time() < self._token_expires_at - 300:
            return
        if not self._refresh_token:
            self.logger.warning("[YouTube] Token 快過期且無 refresh token，請重新登入")
            return
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        async with self._token_lock:
            # 取得 lock 後再檢查一次：等待期間可能已被其他呼叫者刷新
            if time.time() < self._token_expires_at - 300:
                return
            await self._refresh_access_token()

    async def _refresh_access_token(self) -> None:
        """以 refresh_token 換取新的 access_token 並持久化。"""
        # TV mode 用內建公開憑證；custom mode 用使用者提供的憑證
        if self._auth_mode == "tv":
            cid     = YOUTUBE_TV_CLIENT_ID
            csecret = YOUTUBE_TV_CLIENT_SECRET
        elif self._client_id and self._client_secret:
            cid     = self._client_id
            csecret = self._client_secret
        else:
            self.logger.warning("[YouTube] 無法刷新 token：缺少 Client ID/Secret，請重新登入")
            self._refresh_token = ""
            return

        payload = {
            "client_id":     cid,
            "client_secret": csecret,
            "refresh_token": self._refresh_token,
            "grant_type":    "refresh_token",
        }
        try:
            async with self._session.post(OAUTH2_TOKEN_URL, data=payload) as resp:
                result = await resp.json()

            if "access_token" not in result:
                self.logger.error(f"[YouTube] Token 刷新失敗: {result}")
                return

            self._access_token     = result["access_token"]
            self._token_expires_at = time.time() + result.get("expires_in", 3600)
            if "refresh_token" in result:
                self._refresh_token = result["refresh_token"]

            # 一次性更新所有 token 相關欄位
            values_to_save = {
                CONF_AUTH_TOKEN:  self._access_token,
                CONF_EXPIRY_TIME: str(self._token_expires_at),
            }
            if "refresh_token" in result:
                values_to_save[CONF_REFRESH_TOKEN] = self._refresh_token

            await self.mass.config.save_provider_config(
                provider_domain=self.domain,
                values=values_to_save,
                instance_id=self.instance_id,
            )

            self.logger.info("[YouTube] Access token 已自動刷新")

        except Exception as exc:
            self.logger.error(f"[YouTube] Token 刷新例外: {exc}")

    # ------------------------------------------------------------------
    # yt-dlp 認證參數
    # ------------------------------------------------------------------

    def _auth_args(self) -> list[str]:
        """組建 yt-dlp 的認證參數。
        
        兩種方式可並存：
        - OAuth token：用於串流，避免限速（以 Authorization header 注入）
        - Cookies：用於存取個人頁面（playlist 列表等）
        
        若兩者都有，同時傳入，yt-dlp 會優先使用 cookies 認證頁面，
        OAuth header 負責串流品質。
        """
        args = []
        if self._cookies_file:
            args += ["--cookies", self._cookies_file]
        if self._access_token:
            args += ["--add-header", f"Authorization:{self._token_type} {self._access_token}"]
        return args

    def _bgutil_args(self) -> list[str]:
        """若設定了 bgutil PO token server，回傳對應的 --extractor-args 參數。
        bgutil-ytdlp-pot-provider 透過此取得 Proof of Origin Token。

        正確格式（v1.0.0+）：
        - 預設 URL (http://127.0.0.1:4416)：plugin 自動生效，不需額外參數
        - 自訂 URL：--extractor-args "youtubepot-bgutilhttp:base_url=http://host:port"
        """
        if not self._bgutil_url:
            return []
        # 若用戶填的就是預設 URL，不需要傳參數
        default_urls = {"http://127.0.0.1:4416", "http://localhost:4416"}
        if self._bgutil_url in default_urls:
            return []
        return [
            "--extractor-args",
            f"youtubepot-bgutilhttp:base_url={self._bgutil_url}",
        ]

    def _ejs_args(self) -> list[str]:
        """回傳 yt-dlp EJS（External JS）相關參數。
        新版 yt-dlp 需要 JS runtime 來解 n challenge / signature challenge。
        優先使用 deno，找不到則略過（讓 yt-dlp 自行嘗試）。
        """
        import shutil
        args = []
        deno_path = shutil.which("deno")
        if deno_path:
            args += ["--js-runtimes", f"deno:{deno_path}"]
            args += ["--remote-components", "ejs:github"]
        return args

    def _load_youtube_cookies(self) -> dict[str, str]:
        """解析 cookies.txt（Netscape 格式），回傳 youtube.com 的 cookie dict。"""
        if not self._cookies_file:
            return {}
        # 以檔案 mtime 做快取，避免每次 InnerTube WEB 請求都在事件迴圈上重新讀檔
        try:
            mtime = os.path.getmtime(self._cookies_file)
        except OSError:
            return {}
        if self._cookies_cache is not None and self._cookies_cache[0] == mtime:
            return self._cookies_cache[1]
        cookies: dict[str, str] = {}
        try:
            with open(self._cookies_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 7:
                        continue
                    domain, _flag, _path, _secure, _expiry, name, value = parts[:7]
                    if "youtube.com" in domain:
                        cookies[name] = value
        except Exception as exc:
            self.logger.debug(f"[YouTube] 解析 cookies.txt 失敗: {exc}")
        self._cookies_cache = (mtime, cookies)
        return cookies

    def _innertube_cookie_headers(self) -> dict[str, str]:
        """回傳 InnerTube 請求所需的 Cookie + Authorization (SAPISIDHASH) headers。"""
        import hashlib, time
        cookies = self._load_youtube_cookies()
        if not cookies:
            return {}
        cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
        headers: dict[str, str] = {"Cookie": cookie_str}
        sapisid = cookies.get("SAPISID") or cookies.get("__Secure-3PAPISID", "")
        if sapisid:
            ts     = int(time.time())
            origin = "https://www.youtube.com"
            digest = hashlib.sha1(f"{ts} {sapisid} {origin}".encode()).hexdigest()
            headers["Authorization"]  = f"SAPISIDHASH {ts}_{digest}"
            headers["X-Goog-AuthUser"] = "0"
            headers["X-Origin"]        = origin
        return headers

    # ------------------------------------------------------------------
    # Similar Tracks / Recommendations（via ytmusicapi）
    # ------------------------------------------------------------------

    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """透過 ytmusicapi get_watch_playlist(radio=True) 取得相似曲目。"""
        if not self._ytmusic:
            return []
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._ytmusic.get_watch_playlist(
                    videoId=prov_track_id, limit=limit, radio=True,
                ),
            )
        except Exception as exc:
            self.logger.warning(f"[YouTube] get_similar_tracks 失敗: {exc}")
            return []

        tracks: list[Track] = []
        for item in data.get("tracks", []):
            vid = item.get("videoId")
            if not vid or vid == prov_track_id:
                continue
            try:
                track = self._ytmusic_track_to_ma(item)
                tracks.append(track)
            except Exception as exc:
                self.logger.debug(f"[YouTube] 轉換 similar track 失敗: {exc}")
            if len(tracks) >= limit:
                break
        self.logger.debug(f"[YouTube] get_similar_tracks({prov_track_id}) → {len(tracks)} 首")
        return tracks

    async def recommendations(self) -> list[RecommendationFolder]:
        """透過 ytmusicapi get_home() 取得個人化推薦，優先從快取回傳。"""
        if not self._ytmusic:
            return []
        # 快取仍有效則直接回傳
        if self._recommendation_cache is not None:
            self.logger.debug(
                f"[YouTube] recommendations → 從快取回傳 {len(self._recommendation_cache)} 個分類"
            )
            return self._recommendation_cache
        # 快取不存在（首次或已過期）→ 即時取得並存入快取
        return await self._fetch_recommendations()

    async def _sync_recommendations(self) -> None:
        """背景工作：刷新推薦快取（由 register_scheduled_task 定期呼叫）。"""
        self.logger.info("[YouTube] 刷新推薦快取")
        await self._fetch_recommendations()

    async def _fetch_recommendations(self) -> list[RecommendationFolder]:
        """向 YouTube Music 取得推薦並更新快取，回傳結果。"""
        try:
            sections = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._ytmusic.get_home(limit=6),
            )
        except Exception as exc:
            self.logger.warning(f"[YouTube] recommendations (get_home) 失敗: {exc}")
            # 取得失敗時，若有舊快取仍回傳舊的
            return self._recommendation_cache or []

        folders: list[RecommendationFolder] = []
        for idx, section in enumerate(sections or []):
            title = section.get("title", f"推薦 {idx + 1}")
            items: list = []
            for entry in section.get("contents", []):
                try:
                    ma_item = self._ytmusic_home_entry_to_ma(entry)
                    if ma_item:
                        items.append(ma_item)
                except Exception as exc:
                    self.logger.debug(f"[YouTube] 轉換 home entry 失敗: {exc}")
            if not items:
                continue
            folder_id = f"ytm_home_{idx}"
            folders.append(
                RecommendationFolder(
                    item_id=folder_id,
                    provider=self.instance_id,
                    name=title,
                    items=items,
                    is_playable=True,
                )
            )
        self._recommendation_cache = folders
        self._recommendation_cache_at = time.time()
        self.logger.info(f"[YouTube] 推薦快取已更新：{len(folders)} 個分類")
        return folders

    def _ytmusic_best_thumb(self, thumbnails: list[dict]) -> str:
        """從 ytmusicapi 的 thumbnail 列表中取最高解析度的 URL。"""
        best_url = ""
        best_w = 0
        for t in thumbnails:
            if not isinstance(t, dict):
                continue
            url = t.get("url", "")
            if not url:
                continue
            w = t.get("width", 0) or 0
            if w > best_w or not best_url:
                best_url = url
                best_w = w
        return best_url

    def _ytmusic_normalize_artist_browse_id(self, browse_id: str) -> str:
        """與 ytmusicapi.get_artist 相同：MPLA 前綴為 playlist 聚合 id。"""
        bid = browse_id
        if bid.startswith("MPLA"):
            bid = bid[4:]
        return bid

    def _ytmusic_track_to_ma(self, item: dict) -> Track:
        """將 ytmusicapi track dict 轉換為 MA Track 物件。"""
        item = dict(item)
        if not item.get("artists") and item.get("artist"):
            item["artists"] = [{"name": str(item["artist"]), "id": ""}]

        vid = item["videoId"]
        title = item.get("title", vid)

        artists: UniqueList[Artist | ItemMapping] = UniqueList()
        for a in item.get("artists") or []:
            artist_id = a.get("id") or a.get("name", "unknown")
            artists.append(
                ItemMapping(
                    media_type=MediaType.ARTIST,
                    item_id=artist_id,
                    provider=self.instance_id,
                    name=a.get("name", ""),
                )
            )

        thumbs = item.get("thumbnail") or item.get("thumbnails") or []
        if isinstance(thumbs, dict):
            thumbs = thumbs.get("thumbnails", [])
        thumb_url = self._ytmusic_best_thumb(thumbs)

        duration = self._parse_ytm_duration(item.get("length", item.get("duration", "")))
        if not duration and item.get("duration_seconds") is not None:
            try:
                duration = int(item["duration_seconds"])
            except (TypeError, ValueError):
                duration = 0

        track = Track(
            item_id=vid,
            provider=self.domain,
            name=title,
            duration=duration,
            artists=artists,
            provider_mappings={
                ProviderMapping(
                    item_id=vid,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"https://music.youtube.com/watch?v={vid}",
                )
            },
        )
        if thumb_url:
            track.metadata.images = UniqueList(
                [MediaItemImage(
                    type=ImageType.THUMB,
                    path=thumb_url,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )]
            )
        return track

    def _ytmusic_home_entry_to_ma(self, entry: dict) -> Track | ItemMapping | None:
        """將 get_home() 回傳的單筆 entry 轉換為 MA 物件。

        get_home 回傳的 entry 可能是 song / video / album / playlist / artist 等，
        這裡優先處理 song/video（有 videoId），其餘用 ItemMapping 包裝。
        """
        vid = entry.get("videoId")
        if vid:
            return self._ytmusic_track_to_ma(entry)

        thumbs = entry.get("thumbnails") or entry.get("thumbnail") or []
        if isinstance(thumbs, dict):
            thumbs = thumbs.get("thumbnails", [])
        thumb_url = self._ytmusic_best_thumb(thumbs)
        image = (
            MediaItemImage(
                type=ImageType.THUMB,
                path=thumb_url,
                provider=self.instance_id,
                remotely_accessible=True,
            )
            if thumb_url
            else None
        )

        playlist_id = entry.get("playlistId")
        if playlist_id:
            mapping = ItemMapping(
                media_type=MediaType.PLAYLIST,
                item_id=playlist_id,
                provider=self.instance_id,
                name=entry.get("title", playlist_id),
            )
            if image:
                mapping.image = image
            return mapping

        browse_id = entry.get("browseId")
        if browse_id:
            if browse_id.startswith("MPRE"):
                mapping = ItemMapping(
                    media_type=MediaType.ALBUM,
                    item_id=browse_id,
                    provider=self.instance_id,
                    name=entry.get("title", browse_id),
                )
                if image:
                    mapping.image = image
                return mapping
            elif browse_id.startswith("UC"):
                mapping = ItemMapping(
                    media_type=MediaType.ARTIST,
                    item_id=browse_id,
                    provider=self.instance_id,
                    name=entry.get("title", browse_id),
                )
                if image:
                    mapping.image = image
                return mapping
        return None

    @staticmethod
    def _parse_ytm_duration(length_str: str) -> int:
        """解析 ytmusicapi 回傳的時長字串（如 '3:07' 或 '1:02:30'）為秒數。"""
        if not length_str:
            return 0
        parts = length_str.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return int(parts[0])
        except (ValueError, IndexError):
            return 0

    # ------------------------------------------------------------------
    # 搜尋（YouTube InnerTube／yt-dlp + 可選 ytmusicapi 合併）
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_search_tracks(
        ytm_first: list[Track], yt_second: list[Track], limit: int
    ) -> list[Track]:
        """合併曲目：優先 YTMusic，再補 YouTube，依 videoId 去重。"""
        seen: set[str] = set()
        out: list[Track] = []
        for t in ytm_first:
            if t.item_id in seen:
                continue
            seen.add(t.item_id)
            out.append(t)
            if len(out) >= limit:
                return out
        for t in yt_second:
            if t.item_id in seen:
                continue
            seen.add(t.item_id)
            out.append(t)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _merge_search_artists(
        ytm_first: list[Artist], yt_second: list[Artist], limit: int
    ) -> list[Artist]:
        """合併藝人：優先 YTMusic，再補 YouTube 頻道，依 item_id 去重。"""
        seen: set[str] = set()
        out: list[Artist] = []
        for a in ytm_first:
            if a.item_id in seen:
                continue
            seen.add(a.item_id)
            out.append(a)
            if len(out) >= limit:
                return out
        for a in yt_second:
            if a.item_id in seen:
                continue
            seen.add(a.item_id)
            out.append(a)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _merge_search_playlists(
        ytm_first: list[Playlist], yt_second: list[Playlist], limit: int
    ) -> list[Playlist]:
        """合併播放清單：優先 YTMusic，再補 YouTube，依 item_id 去重。"""
        seen: set[str] = set()
        out: list[Playlist] = []
        for p in ytm_first:
            if p.item_id in seen:
                continue
            seen.add(p.item_id)
            out.append(p)
            if len(out) >= limit:
                return out
        for p in yt_second:
            if p.item_id in seen:
                continue
            seen.add(p.item_id)
            out.append(p)
            if len(out) >= limit:
                break
        return out

    async def _search_tracks_via_ytmusic(self, query: str, limit: int) -> list[Track]:
        """ytmusicapi：songs + videos 篩選搜尋（YouTube Music 目錄）。"""
        if not self._ytmusic:
            raise RuntimeError("ytmusicapi 未初始化")
        loop = asyncio.get_event_loop()

        def _sync() -> list[Track]:
            tracks: list[Track] = []
            seen: set[str] = set()
            for flt in ("songs", "videos"):
                try:
                    batch = self._ytmusic.search(query, filter=flt, limit=limit)
                except Exception:
                    batch = []
                for item in batch:
                    vid = item.get("videoId")
                    if not vid or vid in seen:
                        continue
                    rt = item.get("resultType")
                    if rt not in ("song", "video"):
                        continue
                    try:
                        tracks.append(self._ytmusic_track_to_ma(item))
                        seen.add(vid)
                    except Exception as exc:
                        self.logger.debug(f"[YouTube] ytmusic 搜尋轉 Track 失敗: {exc}")
                    if len(tracks) >= limit:
                        return tracks
            return tracks

        return await loop.run_in_executor(None, _sync)

    async def _search_artists_via_ytmusic(self, query: str, limit: int) -> list[Artist]:
        """ytmusicapi：artists 篩選搜尋。"""
        if not self._ytmusic:
            raise RuntimeError("ytmusicapi 未初始化")
        loop = asyncio.get_event_loop()

        def _sync() -> list[Artist]:
            try:
                batch = self._ytmusic.search(query, filter="artists", limit=limit)
            except Exception:
                return []
            artists: list[Artist] = []
            for item in batch:
                if item.get("resultType") != "artist":
                    continue
                bid = item.get("browseId")
                if not bid:
                    continue
                name = item.get("artist") or item.get("title") or bid
                thumbs = item.get("thumbnails") or []
                thumb_url = self._ytmusic_best_thumb(thumbs)
                mapping = ProviderMapping(
                    item_id=bid,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=f"https://music.youtube.com/channel/{bid}",
                )
                artist = Artist(
                    item_id=bid,
                    name=name,
                    provider=self.instance_id,
                    provider_mappings={mapping},
                )
                if thumb_url:
                    artist.metadata.images = [
                        MediaItemImage(
                            type=ImageType.THUMB,
                            path=thumb_url,
                            provider=self.instance_id,
                            remotely_accessible=True,
                        )
                    ]
                artists.append(artist)
                if len(artists) >= limit:
                    break
            return artists

        return await loop.run_in_executor(None, _sync)

    async def _search_playlists_via_ytmusic(self, query: str, limit: int) -> list[Playlist]:
        """ytmusicapi：playlists / community / featured 搜尋後合併。"""
        if not self._ytmusic:
            raise RuntimeError("ytmusicapi 未初始化")
        loop = asyncio.get_event_loop()

        def _sync() -> list[Playlist]:
            out: list[Playlist] = []
            seen: set[str] = set()
            for flt in ("playlists", "community_playlists", "featured_playlists"):
                try:
                    batch = self._ytmusic.search(query, filter=flt, limit=limit)
                except Exception:
                    continue
                for item in batch:
                    if item.get("resultType") != "playlist":
                        continue
                    browse_id = item.get("browseId") or ""
                    pl_id = item.get("playlistId")
                    if not pl_id and browse_id.startswith("VL"):
                        pl_id = browse_id[2:]
                    elif not pl_id:
                        pl_id = browse_id
                    if not pl_id or pl_id in seen:
                        continue
                    seen.add(pl_id)
                    title = item.get("title", pl_id)
                    thumbs = item.get("thumbnails") or []
                    thumb_url = self._ytmusic_best_thumb(thumbs)
                    mapping = ProviderMapping(
                        item_id=pl_id,
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                        url=f"https://www.youtube.com/playlist?list={pl_id}",
                    )
                    pl = Playlist(
                        item_id=pl_id,
                        name=title,
                        provider=self.instance_id,
                        provider_mappings={mapping},
                    )
                    if thumb_url:
                        pl.metadata.images = [
                            MediaItemImage(
                                type=ImageType.THUMB,
                                path=thumb_url,
                                provider=self.instance_id,
                                remotely_accessible=True,
                            )
                        ]
                    out.append(pl)
                    if len(out) >= limit:
                        return out
            return out

        return await loop.run_in_executor(None, _sync)

    async def _search_albums_via_ytmusic(self, query: str, limit: int) -> list[Album]:
        """ytmusicapi：albums 篩選（一般 YouTube 搜尋無專輯型別，僅 YTMusic）。"""
        if not self._ytmusic:
            return []
        loop = asyncio.get_event_loop()

        def _sync() -> list[Album]:
            try:
                batch = self._ytmusic.search(query, filter="albums", limit=limit)
            except Exception:
                return []
            albums: list[Album] = []
            for item in batch:
                if item.get("resultType") != "album":
                    continue
                d = dict(item)
                if d.get("artist") and not d.get("artists"):
                    d["artists"] = [{"name": str(d["artist"]), "id": ""}]
                ma = self._ytmusic_catalog_album_to_ma(d)
                if ma:
                    albums.append(ma)
                if len(albums) >= limit:
                    break
            return albums

        return await loop.run_in_executor(None, _sync)

    async def search(
        self, search_query: str, media_types: list[Any], limit: int = 10
    ) -> SearchResults:
        await self._ensure_fresh_token()
        has_ytm = self._ytmusic is not None

        async def _empty() -> list:
            return []

        async def _merged(kind: str, yt_coro, ytm_coro, merge) -> list:
            # YouTube 與 YTMusic 並行查詢後合併（YTMusic 優先），單邊失敗不影響另一邊
            yt_res, ytm_res = await asyncio.gather(
                yt_coro, ytm_coro, return_exceptions=True
            )
            if isinstance(ytm_res, BaseException):
                self.logger.warning(
                    f"[YouTube] ytmusic 搜尋 {kind} 失敗，僅用 YouTube 結果: {ytm_res}"
                )
                ytm_res = []
            if isinstance(yt_res, BaseException):
                self.logger.warning(f"[YouTube] 搜尋 {kind} (YouTube) 失敗: {yt_res}")
                yt_res = []
            return merge(ytm_res, yt_res, limit)

        # 各 media type 的搜尋全部並行執行，大幅縮短整體搜尋延遲
        pending: dict[str, Any] = {}
        if not media_types or MediaType.TRACK in media_types:
            pending["tracks"] = _merged(
                "tracks",
                self._search_tracks(search_query, limit),
                self._search_tracks_via_ytmusic(search_query, limit)
                if has_ytm else _empty(),
                self._merge_search_tracks,
            )
        if not media_types or MediaType.PLAYLIST in media_types:
            pending["playlists"] = _merged(
                "playlists",
                self._search_playlists(search_query, limit),
                self._search_playlists_via_ytmusic(search_query, limit)
                if has_ytm else _empty(),
                self._merge_search_playlists,
            )
        if not media_types or MediaType.ARTIST in media_types:
            pending["artists"] = _merged(
                "artists",
                self._search_artists(search_query, limit),
                self._search_artists_via_ytmusic(search_query, limit)
                if has_ytm else _empty(),
                self._merge_search_artists,
            )
        # 專輯：僅 YTMusic 有結構化結果
        if (not media_types or MediaType.ALBUM in media_types) and has_ytm:
            pending["albums"] = _merged(
                "albums",
                _empty(),
                self._search_albums_via_ytmusic(search_query, limit),
                lambda ytm, _yt, lim: ytm[:lim],
            )

        gathered = await asyncio.gather(*pending.values()) if pending else []
        results = dict(zip(pending.keys(), gathered))
        tracks: list[Track] = results.get("tracks", [])
        playlists: list[Playlist] = results.get("playlists", [])
        artists: list[Artist] = results.get("artists", [])
        albums: list[Album] = results.get("albums", [])

        self.logger.debug(
            f"[YouTube] 搜尋「{search_query}」→ "
            f"{len(tracks)} tracks, {len(playlists)} playlists, "
            f"{len(artists)} artists, {len(albums)} albums"
        )
        return SearchResults(
            tracks=tracks,
            artists=artists,
            albums=albums,
            playlists=playlists,
            radio=[],
        )

    async def _search_tracks(self, query: str, limit: int) -> list[Track]:
        """搜尋影片。InnerTube → yt-dlp（搜尋每次耗 100 units，不用 Data API 以保留配額）。"""
        return await self._try_methods("搜尋 tracks", [
            ("InnerTube", self._search_tracks_via_innertube),
            ("yt-dlp",    self._search_tracks_via_ytdlp),
        ], query, limit)

    # ------------------------------------------------------------------
    # InnerTube API 方法（YouTube TV 模式）
    # ------------------------------------------------------------------

    @staticmethod
    def _innertube_extract_text(renderer: dict) -> str:
        """從 InnerTube renderer 提取文字。"""
        if not renderer:
            return ""
        if "simpleText" in renderer:
            return renderer["simpleText"]
        if "runs" in renderer:
            return "".join(r.get("text", "") for r in renderer["runs"])
        return ""

    @staticmethod
    def _innertube_find_all(data: any, *keys: str) -> list:
        """遞迴找出 InnerTube response 中所有指定 key(s) 的 renderer。

        支援多個 key 名稱（按優先順序），例如同時找
        videoRenderer 和 compactVideoRenderer。
        """
        results = []
        if isinstance(data, dict):
            for k in keys:
                if k in data:
                    results.append(data[k])
            for v in data.values():
                results.extend(YouTubeProvider._innertube_find_all(v, *keys))
        elif isinstance(data, list):
            for item in data:
                results.extend(YouTubeProvider._innertube_find_all(item, *keys))
        return results

    @staticmethod
    def _innertube_collect_renderer_keys(data: any, depth: int = 0, max_depth: int = 6) -> set[str]:
        """遞迴收集 response 中所有以 Renderer 結尾的 key（用於 debug 診斷）。"""
        keys: set[str] = set()
        if depth > max_depth:
            return keys
        if isinstance(data, dict):
            for k, v in data.items():
                if k.endswith("Renderer"):
                    keys.add(k)
                keys.update(YouTubeProvider._innertube_collect_renderer_keys(v, depth + 1, max_depth))
        elif isinstance(data, list):
            for item in data:
                keys.update(YouTubeProvider._innertube_collect_renderer_keys(item, depth + 1, max_depth))
        return keys

    async def _search_tracks_via_innertube(self, query: str, limit: int) -> list[Track]:
        """透過 InnerTube 搜尋影片。"""
        data  = await self._innertube_request(
            "search", {"query": query, "params": "EgIQAQ=="},
            use_web_client=True,
        )
        # EgIQAQ== = filter: videos only
        items = self._innertube_find_all(
            data, "videoRenderer", "compactVideoRenderer",
        )
        if not items:
            renderer_keys = self._innertube_collect_renderer_keys(data)
            self.logger.warning(
                f"[YouTube] InnerTube search tracks: query={query!r} "
                f"找不到 videoRenderer，response 中的 renderer 類型: "
                f"{sorted(renderer_keys)}"
            )
        else:
            self.logger.debug(
                f"[YouTube] InnerTube search tracks: query={query!r} "
                f"videoRenderer 數量={len(items)}"
            )
        tracks: list[Track] = []
        for item in items[:limit]:
            vid_id = item.get("videoId")
            if not vid_id:
                continue
            title = self._innertube_extract_text(item.get("title", {}))
            channel = item.get("ownerText") or item.get("longBylineText") or {}
            channel_name = self._normalize_youtube_topic_channel_name(
                self._innertube_extract_text(channel)
            )
            channel_id   = ""
            for run in (channel.get("runs") or []):
                ep = run.get("navigationEndpoint", {})
                channel_id = (
                    ep.get("browseEndpoint", {}).get("browseId", "")
                    or ep.get("commandMetadata", {}).get("webCommandMetadata", {}).get("url", "")
                )
                if channel_id.startswith("UC"):
                    break
            # duration
            duration = 0
            dur_text = self._innertube_extract_text(item.get("lengthText", {}))
            if dur_text:
                import re
                parts = re.split(r":", dur_text)
                try:
                    if len(parts) == 3:
                        duration = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                    elif len(parts) == 2:
                        duration = int(parts[0])*60 + int(parts[1])
                except Exception:
                    pass
            # thumbnail
            thumbs = item.get("thumbnail", {}).get("thumbnails", [])
            thumb_url = thumbs[-1]["url"] if thumbs else ""

            artist_mapping = ProviderMapping(
                item_id=channel_id or vid_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
                url=f"https://www.youtube.com/channel/{channel_id}" if channel_id else "",
            )
            artist = Artist(
                item_id=channel_id or vid_id, name=channel_name or "YouTube",
                provider=self.instance_id, provider_mappings={artist_mapping},
            )
            track_mapping = ProviderMapping(
                item_id=vid_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
            )
            track = Track(
                item_id=vid_id, name=title, artists=[artist],
                provider=self.instance_id, duration=duration,
                provider_mappings={track_mapping},
            )
            if thumb_url:
                track.metadata.images = [MediaItemImage(
                    type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
                )]
            tracks.append(track)
        self.logger.debug(f"[YouTube] InnerTube 搜尋 tracks 共 {len(tracks)} 筆")
        return tracks

    async def _search_playlists_via_innertube(self, query: str, limit: int) -> list[Playlist]:
        """透過 InnerTube 搜尋播放清單。"""
        data  = await self._innertube_request(
            "search", {"query": query, "params": "EgIQAw=="},
            use_web_client=True,
        )
        # EgIQAw== = filter: playlists only
        items = self._innertube_find_all(
            data, "playlistRenderer", "compactPlaylistRenderer",
        )
        if not items:
            renderer_keys = self._innertube_collect_renderer_keys(data)
            self.logger.warning(
                f"[YouTube] InnerTube search playlists: query={query!r} "
                f"找不到 playlistRenderer，renderer 類型: {sorted(renderer_keys)}"
            )
        else:
            self.logger.debug(
                f"[YouTube] InnerTube search playlists: query={query!r} "
                f"playlistRenderer={len(items)}"
            )
        playlists: list[Playlist] = []
        for item in items[:limit]:
            pl_id = item.get("playlistId")
            if not pl_id:
                continue
            title    = self._innertube_extract_text(item.get("title", {}))
            thumbs   = item.get("thumbnails", [{}])
            thumbs   = thumbs[0].get("thumbnails", []) if thumbs else []
            thumb_url = thumbs[-1]["url"] if thumbs else ""
            mapping  = ProviderMapping(
                item_id=pl_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
                url=f"https://www.youtube.com/playlist?list={pl_id}",
            )
            pl = Playlist(
                item_id=pl_id, name=title or pl_id,
                provider=self.instance_id, provider_mappings={mapping},
            )
            if thumb_url:
                pl.metadata.images = [MediaItemImage(
                    type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
                )]
            playlists.append(pl)
        self.logger.debug(f"[YouTube] InnerTube 搜尋 playlists 共 {len(playlists)} 筆")
        return playlists

    async def _search_artists_via_innertube(self, query: str, limit: int) -> list[Artist]:
        """透過 InnerTube 搜尋頻道。"""
        data  = await self._innertube_request(
            "search", {"query": query, "params": "EgIQAg=="},
            use_web_client=True,
        )
        # EgIQAg== = filter: channels only
        items = self._innertube_find_all(
            data, "channelRenderer", "compactChannelRenderer",
        )
        if not items:
            renderer_keys = self._innertube_collect_renderer_keys(data)
            self.logger.warning(
                f"[YouTube] InnerTube search artists: query={query!r} "
                f"找不到 channelRenderer，renderer 類型: {sorted(renderer_keys)}"
            )
        artists: list[Artist] = []
        for item in items[:limit]:
            channel_id = item.get("channelId")
            if not channel_id:
                continue
            name = self._normalize_youtube_topic_channel_name(
                self._innertube_extract_text(item.get("title", {}))
            )
            thumbs = item.get("thumbnail", {}).get("thumbnails", [])
            thumb_url = thumbs[-1]["url"] if thumbs else ""
            if thumb_url.startswith("//"):
                thumb_url = "https:" + thumb_url
            mapping = ProviderMapping(
                item_id=channel_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
                url=f"https://www.youtube.com/channel/{channel_id}",
            )
            artist = Artist(
                item_id=channel_id, name=name or channel_id,
                provider=self.instance_id, provider_mappings={mapping},
            )
            if thumb_url:
                artist.metadata.images = [MediaItemImage(
                    type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
                )]
            artists.append(artist)
        self.logger.debug(f"[YouTube] InnerTube 搜尋 artists 共 {len(artists)} 筆")
        return artists

    async def _get_artist_via_innertube(self, channel_id: str) -> Artist:
        """透過 InnerTube 取得頻道資訊。"""
        data = await self._innertube_request("browse", {"browseId": channel_id})
        header = data.get("header", {})

        # TV client 用 pageHeaderRenderer，一般 client 用 c4TabbedHeaderRenderer
        c4   = header.get("c4TabbedHeaderRenderer", {})
        page = header.get("pageHeaderRenderer", {})

        name = ""
        thumbs = []

        if c4:
            name   = self._innertube_extract_text(c4.get("title", {}))
            thumbs = c4.get("avatar", {}).get("thumbnails", [])
        elif page:
            name   = self._innertube_extract_text(page.get("pageTitle", {}))
            vm     = page.get("content", {}).get("pageHeaderViewModel", {})
            thumbs = (
                vm.get("image", {}).get("decoratedAvatarViewModel", {})
                  .get("avatar", {}).get("avatarViewModel", {})
                  .get("image", {}).get("sources", [])
                or vm.get("image", {}).get("thumbnails", [])
            )

        # TV client 不回傳頻道名稱，改用 yt-dlp 取
        if not name:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "yt-dlp", "--quiet", "--no-warnings",
                    "--dump-single-json", "--playlist-items", "0",
                    f"https://www.youtube.com/channel/{channel_id}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if stdout:
                    info = json.loads(stdout.decode())
                    name   = info.get("channel") or info.get("uploader") or ""
                    yt_thumbs = info.get("thumbnails", [])
                    if yt_thumbs:
                        thumbs = [{"url": yt_thumbs[-1]["url"]}]
                    self.logger.debug(f"[YouTube] yt-dlp 取頻道名稱: {name!r}")
            except Exception as exc:
                self.logger.debug(f"[YouTube] yt-dlp 取頻道名稱失敗: {exc}")

        name = self._normalize_youtube_topic_channel_name(name or channel_id)
        thumb_url = thumbs[-1]["url"] if thumbs else ""
        if thumb_url.startswith("//"):
            thumb_url = "https:" + thumb_url

        self.logger.debug(f"[YouTube] _get_artist_via_innertube: {channel_id} name={name!r}")

        mapping = ProviderMapping(
            item_id=channel_id, provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/channel/{channel_id}",
        )
        artist = Artist(
            item_id=channel_id, name=name,
            provider=self.instance_id, provider_mappings={mapping},
        )
        if thumb_url:
            artist.metadata.images = [MediaItemImage(
                type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
            )]
        return artist

    def _parse_duration_text(self, dur_text: str) -> int:
        """將 HH:MM:SS 或 MM:SS 字串轉為秒數。"""
        import re
        parts = re.split(r":", dur_text)
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            pass
        return 0

    def _innertube_items_to_tracks(
        self, items: list, channel_id: str, channel_name: str = ""
    ) -> list[Track]:
        """將 InnerTube videoRenderer / gridVideoRenderer 清單轉為 Track 列表。"""
        tracks: list[Track] = []
        for item in items:
            vid_id = item.get("videoId")
            if not vid_id:
                continue
            raw_title = self._innertube_extract_text(item.get("title", {}))
            dur_text  = self._innertube_extract_text(item.get("lengthText", {}))
            duration  = self._parse_duration_text(dur_text) if dur_text else 0
            thumbs    = item.get("thumbnail", {}).get("thumbnails", [])
            thumb_url = thumbs[-1]["url"] if thumbs else ""

            # 頻道名稱：優先從 item 取，fallback 到傳入值
            ch_name = (
                self._innertube_extract_text(item.get("ownerText", {}))
                or self._innertube_extract_text(item.get("shortBylineText", {}))
                or channel_name or channel_id
            )

            parsed_title, parsed_artist = self._parse_title(raw_title, ch_name)
            track_name  = parsed_title or raw_title or vid_id
            artist_name = parsed_artist or self._normalize_youtube_topic_channel_name(
                ch_name
            )

            artist_mapping = ProviderMapping(
                item_id=channel_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
            )
            artist = Artist(
                item_id=channel_id, name=artist_name,
                provider=self.instance_id, provider_mappings={artist_mapping},
            )
            track_mapping = ProviderMapping(
                item_id=vid_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
            )
            track = Track(
                item_id=vid_id, name=track_name, artists=[artist],
                provider=self.instance_id, duration=duration,
                provider_mappings={track_mapping},
            )
            if thumb_url:
                track.metadata.images = [MediaItemImage(
                    type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
                )]
            tracks.append(track)
        return tracks

    async def _get_channel_videos_via_innertube(
        self, channel_id: str, max_videos: int = 100
    ) -> list[Track]:
        """透過 InnerTube 取得頻道影片列表，支援分頁（預設最多 500 支）。"""

        def _extract_items_and_token(data: dict) -> tuple[list, str]:
            # richVideoRenderer（WEB client channel tab）和 videoRenderer / gridVideoRenderer
            # 均直接遞迴搜尋，_innertube_find_all 會穿透 richItemRenderer 包層
            items = (
                self._innertube_find_all(data, "videoRenderer")
                + self._innertube_find_all(data, "gridVideoRenderer")
                + self._innertube_find_all(data, "richVideoRenderer")
            )

            # continuation token
            next_token = ""
            for ci in self._innertube_find_all(data, "continuationCommand"):
                token = ci.get("token") or ci.get("continuation", "")
                if token:
                    next_token = token
                    break
            if not next_token:
                for ci in self._innertube_find_all(data, "continuationItemRenderer"):
                    token = (
                        ci.get("continuationEndpoint", {})
                        .get("continuationCommand", {})
                        .get("token", "")
                    )
                    if token:
                        next_token = token
                        break
            return items, next_token

        # 第一頁：頻道 Videos tab（EgZ2aWRlb3M%3D），需用 WEB client
        data = await self._innertube_request(
            "browse", {"browseId": channel_id, "params": "EgZ2aWRlb3M%3D"},
            use_web_client=True,
        )
        # 嘗試從 header 取頻道名稱
        channel_name = ""
        for key in ("header", "metadata"):
            hdr = data.get(key, {})
            for sub in hdr.values() if isinstance(hdr, dict) else []:
                if isinstance(sub, dict):
                    channel_name = (
                        self._innertube_extract_text(sub.get("title", {}))
                        or sub.get("title", "")
                    )
                    if channel_name:
                        break
            if channel_name:
                break

        all_items, next_token = _extract_items_and_token(data)
        page = 0
        while next_token and len(all_items) < max_videos and page < 50:
            cont_data = await self._innertube_request(
                "browse", {"continuation": next_token}, use_web_client=True,
            )
            new_items, next_token = _extract_items_and_token(cont_data)
            if not new_items:
                break
            all_items.extend(new_items)
            page += 1
            self.logger.debug(
                f"[YouTube] InnerTube 頻道分頁 {page}: 累計 {len(all_items)} 支"
            )

        # 截斷到 max_videos
        all_items = all_items[:max_videos]
        tracks = self._innertube_items_to_tracks(all_items, channel_id, channel_name)
        self.logger.info(
            f"[YouTube] InnerTube 頻道 {channel_id} 共 {len(tracks)} 支影片"
            + (f"（分 {page} 頁）" if page > 0 else "")
        )
        return tracks

    async def _get_playlist_via_innertube(self, playlist_id: str) -> Playlist:
        """透過 InnerTube 取得播放清單資訊。"""
        data   = await self._innertube_request("browse", {"browseId": f"VL{playlist_id}"})
        header = data.get("header", {})
        pl_header = (
            header.get("playlistHeaderRenderer")
            or header.get("musicDetailHeaderRenderer")
            or {}
        )
        title  = self._innertube_extract_text(pl_header.get("title", {})) or playlist_id
        thumbs = pl_header.get("thumbnail", {}).get("thumbnails", [])
        thumb_url = thumbs[-1]["url"] if thumbs else ""
        mapping = ProviderMapping(
            item_id=playlist_id, provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/playlist?list={playlist_id}",
        )
        pl = Playlist(
            item_id=playlist_id, name=title,
            provider=self.instance_id, provider_mappings={mapping},
        )
        if thumb_url:
            pl.metadata.images = [MediaItemImage(
                type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
            )]
        return pl

    async def _get_playlist_tracks_via_innertube(self, playlist_id: str) -> list[Track]:
        """透過 InnerTube 取得播放清單 tracks。"""
        # PLAOYGtd_ 是個人私人清單，WEB client 需要 cookie session（不支援 Bearer token）
        # 直接 raise 讓呼叫方 fallback 到 yt-dlp+cookies
        if playlist_id.startswith("PLAOYGtd_"):
            raise RuntimeError("PLAOYGtd_ 私人清單不支援 InnerTube，改用 yt-dlp+cookies")
        data  = await self._innertube_request("browse", {"browseId": f"VL{playlist_id}"})

        # TV client 回傳 tileRenderer，一般 client 回傳 playlistVideoRenderer
        # 先嘗試從 TV client 的 twoColumnRenderer 右欄取
        tile_items: list[dict] = []
        try:
            right_col = (data["contents"]["tvBrowseRenderer"]["content"]
                         ["tvSurfaceContentRenderer"]["content"]
                         ["twoColumnRenderer"]["rightColumn"])
            pl_list = right_col.get("playlistVideoListRenderer", {})
            for entry in pl_list.get("contents", []):
                tile = entry.get("tileRenderer")
                if tile:
                    tile_items.append(tile)
        except (KeyError, TypeError):
            pass

        # fallback：一般 client 的 playlistVideoRenderer
        items = self._innertube_find_all(data, "playlistVideoRenderer")

        self.logger.debug(
            f"[YouTube] InnerTube 播放清單 {playlist_id}: "
            f"tileRenderer={len(tile_items)}, playlistVideoRenderer={len(items)}"
        )

        tracks: list[Track] = []

        # 處理 tileRenderer（TV client）
        for tile in tile_items:
            # videoId 在 onSelectCommand 裡
            on_select = tile.get("onSelectCommand", {})
            vid_id = (
                on_select.get("watchEndpoint", {}).get("videoId")
                or on_select.get("watchNextEndpoint", {}).get("videoId")
                or ""
            )
            if not vid_id:
                # 嘗試從 header thumbnail overlay 找
                continue
            header   = tile.get("header", {}).get("tileHeaderRenderer", {})
            metadata = tile.get("metadata", {}).get("tileMetadataRenderer", {})
            raw_title = (
                self._innertube_extract_text(metadata.get("title", {}))
                or self._innertube_extract_text(header.get("title", {}))
                or ""
            )
            # duration 從 thumbnailOverlay 取
            duration = 0
            for overlay in header.get("thumbnailOverlays", []):
                ts = overlay.get("thumbnailOverlayTimeStatusRenderer", {})
                dur_text = self._innertube_extract_text(ts.get("text", {}))
                if dur_text:
                    duration = self._parse_duration_text(dur_text)
                    break
            # 縮圖
            thumbs    = header.get("thumbnail", {}).get("thumbnails", [])
            thumb_url = thumbs[-1]["url"] if thumbs else ""
            # channel name from lines
            ch_name = ""
            for line in metadata.get("lines", []):
                for li in line.get("lineRenderer", {}).get("items", []):
                    text = self._innertube_extract_text(
                        li.get("lineItemRenderer", {}).get("text", {})
                    )
                    if text:
                        ch_name = text
                        break
                if ch_name:
                    break

            parsed_title, parsed_artist = self._parse_title(raw_title, ch_name)
            track_name  = parsed_title or raw_title or vid_id
            artist_name = (
                parsed_artist
                or self._normalize_youtube_topic_channel_name(ch_name)
                or "YouTube"
            )

            artist_mapping = ProviderMapping(
                item_id=vid_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
            )
            artist = Artist(
                item_id=vid_id, name=artist_name,
                provider=self.instance_id, provider_mappings={artist_mapping},
            )
            track_mapping = ProviderMapping(
                item_id=vid_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
            )
            track = Track(
                item_id=vid_id, name=track_name, artists=[artist],
                provider=self.instance_id, duration=duration,
                provider_mappings={track_mapping},
            )
            if thumb_url:
                track.metadata.images = [MediaItemImage(
                    type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
                )]
            tracks.append(track)

        # 如果 tile_items 沒有結果，用原本的 playlistVideoRenderer 邏輯
        if not tracks:
            for item in items:
                vid_id = item.get("videoId")
                if not vid_id:
                    continue
                raw_title = self._innertube_extract_text(item.get("title", {}))
                duration  = 0
                dur_secs  = item.get("lengthSeconds")
                if dur_secs:
                    try:
                        duration = int(dur_secs)
                    except Exception:
                        pass
                channel_id   = item.get("shortBylineText", {})
                channel_runs = channel_id.get("runs", []) if isinstance(channel_id, dict) else []
                ch_id   = ""
                ch_name = ""
                for run in channel_runs:
                    ch_name = run.get("text", "")
                    ep = run.get("navigationEndpoint", {})
                    ch_id = ep.get("browseEndpoint", {}).get("browseId", "")

                parsed_title, parsed_artist = self._parse_title(raw_title, ch_name)
                track_name  = parsed_title or raw_title
                artist_name = (
                    parsed_artist
                    or self._normalize_youtube_topic_channel_name(ch_name)
                    or "YouTube"
                )

                thumbs    = item.get("thumbnail", {}).get("thumbnails", [])
                thumb_url = thumbs[-1]["url"] if thumbs else ""
                artist_mapping = ProviderMapping(
                    item_id=ch_id or vid_id, provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
                artist = Artist(
                    item_id=ch_id or vid_id, name=artist_name,
                    provider=self.instance_id, provider_mappings={artist_mapping},
                )
                track_mapping = ProviderMapping(
                    item_id=vid_id, provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
                track = Track(
                    item_id=vid_id, name=track_name, artists=[artist],
                    provider=self.instance_id, duration=duration,
                    provider_mappings={track_mapping},
                )
                if thumb_url:
                    track.metadata.images = [MediaItemImage(
                        type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
                    )]
                tracks.append(track)
        self.logger.debug(f"[YouTube] InnerTube 播放清單 {playlist_id} 共 {len(tracks)} 首")
        return tracks

    async def _search_tracks_via_api(self, query: str, limit: int) -> list[Track]:
        """用 YouTube Data API v3 搜尋影片。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {"part": "snippet", "q": query, "type": "video",
                   "maxResults": str(limit), "videoCategoryId": "10"}  # 10 = Music
        results: list[Track] = []
        async with self._session.get(
            f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
        ) as resp:
            if resp.status == 401:
                await self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
                ) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()
        for item in data.get("items", []):
            vid_id  = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not vid_id:
                continue
            channel_id   = snippet.get("channelId", "")
            channel_name = snippet.get("channelTitle", "YouTube")
            artist_mapping = ProviderMapping(
                item_id=channel_id or vid_id,
                provider_domain=self.domain, provider_instance=self.instance_id,
                url=f"https://www.youtube.com/channel/{channel_id}" if channel_id else "",
            )
            raw_title = snippet.get("title", "")
            parsed_title, parsed_artist = self._parse_title(raw_title, channel_name)
            track_name   = parsed_title or raw_title
            artist_name  = parsed_artist or self._normalize_youtube_topic_channel_name(
                channel_name
            )
            artist = Artist(item_id=channel_id or vid_id, name=artist_name,
                            provider=self.instance_id, provider_mappings={artist_mapping})
            track_mapping = ProviderMapping(
                item_id=vid_id, provider_domain=self.domain, provider_instance=self.instance_id,
            )
            track = Track(item_id=vid_id, name=track_name,
                          artists=[artist], provider=self.instance_id,
                          provider_mappings={track_mapping})
            thumbnails = snippet.get("thumbnails", {})
            for res in ("high", "medium", "default"):
                if res in thumbnails:
                    track.metadata.images = [MediaItemImage(
                        type=ImageType.THUMB, path=thumbnails[res]["url"],
                        provider=self.instance_id,
                    )]
                    break
            results.append(track)
        return results

    async def _search_tracks_via_ytdlp(self, query: str, limit: int) -> list[Track]:
        """用 yt-dlp 搜尋影片（fallback）。"""
        args = (
            ["yt-dlp", "--quiet", "--no-warnings", "--dump-json",
             "--flat-playlist", "--skip-download"]
            + self._bgutil_args()
            + self._auth_args()
            + [f"ytsearch{limit}:{query}"]
        )
        results: list[Track] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr_data = await proc.communicate()
            if proc.returncode != 0 and not stdout:
                self.logger.error(f"[YouTube] yt-dlp 搜尋 tracks 失敗: {stderr_data.decode().strip()}")
                return results
            for line in stdout.decode().splitlines():
                try:
                    track = self._to_track(json.loads(line))
                    if track:
                        results.append(track)
                except Exception as e:
                    self.logger.warning(f"[YouTube] 解析 track 失敗: {e}")
        except Exception as e:
            self.logger.error(f"[YouTube] _search_tracks_via_ytdlp 例外: {e}", exc_info=True)
        return results

    async def _search_playlists(self, query: str, limit: int) -> list[Playlist]:
        """搜尋播放清單。InnerTube → yt-dlp（不用 Data API 避免配額消耗）。"""
        return await self._try_methods("搜尋 playlists", [
            ("InnerTube", self._search_playlists_via_innertube),
            ("yt-dlp",    self._search_playlists_via_ytdlp),
        ], query, limit)

    async def _search_playlists_via_api(self, query: str, limit: int) -> list[Playlist]:
        """用 YouTube Data API v3 搜尋播放清單。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {"part": "snippet", "q": query, "type": "playlist", "maxResults": str(limit)}
        results: list[Playlist] = []
        async with self._session.get(
            f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
        ) as resp:
            if resp.status == 401:
                await self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
                ) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()
        for item in data.get("items", []):
            pl_id   = item.get("id", {}).get("playlistId")
            snippet = item.get("snippet", {})
            if not pl_id:
                continue
            mapping = ProviderMapping(
                item_id=pl_id, provider_domain=self.domain, provider_instance=self.instance_id,
                url=f"https://www.youtube.com/playlist?list={pl_id}",
            )
            playlist = Playlist(item_id=pl_id, name=snippet.get("title", pl_id),
                                provider=self.instance_id, provider_mappings={mapping},
                                is_editable=False)
            thumbnails = snippet.get("thumbnails", {})
            for res in ("high", "medium", "default"):
                if res in thumbnails:
                    playlist.metadata.images = [MediaItemImage(
                        type=ImageType.THUMB, path=thumbnails[res]["url"],
                        provider=self.instance_id,
                    )]
                    break
            results.append(playlist)
        return results

    async def _search_playlists_via_ytdlp(self, query: str, limit: int) -> list[Playlist]:
        """用 yt-dlp 搜尋播放清單（fallback）。
        
        yt-dlp 的 ytsearch 只回傳影片，playlist 搜尋需要用 YouTube 搜尋頁面。
        URL 格式：https://www.youtube.com/results?search_query=...&sp=EgIQAw%3D%3D
        sp=EgIQAw== 是 YouTube 搜尋過濾器 "播放清單"
        """
        # 用 YouTube 搜尋頁面搜尋播放清單
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        # sp=EgIQAw%3D%3D = 過濾播放清單
        search_url = (
            f"https://www.youtube.com/results"
            f"?search_query={encoded_query}&sp=EgIQAw%3D%3D"
        )
        args = (
            ["yt-dlp", "--quiet", "--no-warnings", "--dump-json",
             "--flat-playlist", "--playlist-end", str(limit)]
            + self._bgutil_args()
            + self._auth_args()
            + [search_url]
        )
        results: list[Playlist] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr_data = await proc.communicate()
            if proc.returncode != 0 and not stdout:
                self.logger.debug(
                    f"[YouTube] yt-dlp playlist 搜尋失敗: {stderr_data.decode()[:200]}"
                )
                return results
            for line in stdout.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 搜尋頁回傳的 playlist entry：有 url 且 url 包含 /playlist?list=
                    url  = data.get("url", "") or data.get("webpage_url", "")
                    ie   = data.get("ie_key", "").lower()
                    pl_id = data.get("id", "")
                    # 過濾：只取播放清單（排除影片）
                    if "playlist?list=" in url or ie in ("youtubeplaylist", "youtube:tab"):
                        if not pl_id:
                            import re
                            m = re.search(r"list=([A-Za-z0-9_-]+)", url)
                            pl_id = m.group(1) if m else ""
                        if not pl_id:
                            continue
                        title = data.get("title", "") or pl_id
                        thumb_url = data.get("thumbnail", "") or ""
                        mapping = ProviderMapping(
                            item_id=pl_id, provider_domain=self.domain,
                            provider_instance=self.instance_id,
                            url=f"https://www.youtube.com/playlist?list={pl_id}",
                        )
                        pl = Playlist(
                            item_id=pl_id, name=title,
                            provider=self.instance_id, provider_mappings={mapping},
                        )
                        if thumb_url:
                            pl.metadata.images = [MediaItemImage(
                                type=ImageType.THUMB, path=thumb_url,
                                provider=self.instance_id,
                            )]
                        results.append(pl)
                        if len(results) >= limit:
                            break
                except Exception as e:
                    self.logger.debug(f"[YouTube] 解析 playlist 搜尋結果失敗: {e}")
            self.logger.debug(
                f"[YouTube] yt-dlp playlist 搜尋 {query!r} 共 {len(results)} 筆"
            )
        except Exception as e:
            self.logger.error(f"[YouTube] _search_playlists_via_ytdlp 例外: {e}", exc_info=True)
        return results

    async def _search_artists(self, query: str, limit: int) -> list[Artist]:
        """搜尋頻道（artists）。InnerTube → yt-dlp（不用 Data API 避免配額消耗）。"""
        return await self._try_methods("搜尋 artists", [
            ("InnerTube", self._search_artists_via_innertube),
            ("yt-dlp",    self._search_artists_via_ytdlp),
        ], query, limit)

    async def _search_artists_via_api(self, query: str, limit: int) -> list[Artist]:
        """用 YouTube Data API v3 搜尋頻道。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {"part": "snippet", "q": query, "type": "channel", "maxResults": str(limit)}
        results: list[Artist] = []
        async with self._session.get(
            f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
        ) as resp:
            if resp.status == 401:
                await self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
                ) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()
        for item in data.get("items", []):
            channel_id   = item.get("id", {}).get("channelId")
            snippet      = item.get("snippet", {})
            channel_name = self._normalize_youtube_topic_channel_name(
                snippet.get("title", "")
            )
            if not channel_id or not channel_name:
                continue
            mapping = ProviderMapping(
                item_id=channel_id, provider_domain=self.domain, provider_instance=self.instance_id,
                url=f"https://www.youtube.com/channel/{channel_id}",
            )
            artist = Artist(item_id=channel_id, name=channel_name,
                            provider=self.instance_id, provider_mappings={mapping})
            thumbnails = snippet.get("thumbnails", {})
            for res in ("high", "medium", "default"):
                if res in thumbnails:
                    artist.metadata.images = [MediaItemImage(
                        type=ImageType.THUMB, path=thumbnails[res]["url"],
                        provider=self.instance_id,
                    )]
                    break
            results.append(artist)
        return results

    async def _search_artists_via_ytdlp(self, query: str, limit: int) -> list[Artist]:
        """用 yt-dlp 搜尋頻道（fallback）。"""
        args = (
            ["yt-dlp", "--quiet", "--no-warnings", "--dump-json",
             "--flat-playlist", "--skip-download"]
            + self._bgutil_args()
            + self._auth_args()
            + [f"ytsearchall{limit}:{query}"]
        )
        results: list[Artist] = []
        seen_channels: set[str] = set()
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr_data = await proc.communicate()
            if proc.returncode != 0 and not stdout:
                return results
            for line in stdout.decode().splitlines():
                try:
                    data = json.loads(line)
                    channel_id   = data.get("channel_id") or data.get("uploader_id")
                    channel_name = data.get("channel") or data.get("uploader")
                    if not channel_id or not channel_name or channel_id in seen_channels:
                        continue
                    seen_channels.add(channel_id)
                    artist = self._channel_to_artist(channel_id, channel_name, data)
                    if artist:
                        results.append(artist)
                        if len(results) >= limit:
                            break
                except Exception as e:
                    self.logger.debug(f"[YouTube] 解析 artist 失敗: {e}")
        except Exception as e:
            self.logger.error(f"[YouTube] _search_artists_via_ytdlp 例外: {e}", exc_info=True)
        return results

    # ------------------------------------------------------------------
    # 媒體物件
    # ------------------------------------------------------------------

    # 頻道名稱如果是這些，代表不是真正的 artist，優先用標題解析
    _GENERIC_CHANNEL_NAMES = {
        "youtube", "kkbox", "kkbox music", "街聲", "five music", "hits fm",
        "hit fm", "rock music", "forward music", "sony music taiwan",
        "sony music", "universal music", "warner music", "emi music",
        "avex taiwan", "avex", "sm entertainment", "yg entertainment",
        "jyp entertainment", "big hit", "hybe", "starship", "cube entertainment",
        "kkbox官方", "官方頻道", "official",
    }

    @staticmethod
    def _normalize_youtube_topic_channel_name(name: str) -> str:
        """移除 YouTube 官方音樂 Topic 頻道常見的「 - Topic」後綴。

        Data API／InnerTube 回傳的 channel／uploader 常為「藝人名 - Topic」，
        非本程式加上；若不處理，會與其他目錄的「The Weeknd」等名稱不一致，
        並可能影響依藝人名比對的顯示（例如 Available on）。
        """
        if not name or not isinstance(name, str):
            return name
        import re
        s = name.strip()
        cleaned = re.sub(r"\s*[-–—]\s*Topic\s*$", "", s, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else s

    @staticmethod
    def _prefer_ytmusic_author_name(ytm_author: str, channel_name: str) -> str:
        """若 YTMusic player 回傳本地化藝人名而頻道／Data API 僅英文，優先採用前者。"""
        if not ytm_author or not isinstance(ytm_author, str):
            return channel_name
        import re
        y = YouTubeProvider._normalize_youtube_topic_channel_name(ytm_author.strip())
        if not y:
            return channel_name
        _cjk = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
        ytm_cjk = bool(_cjk.search(y))
        ch_cjk = bool(_cjk.search(channel_name or ""))
        if ytm_cjk and not ch_cjk:
            return y
        if ytm_cjk and ch_cjk:
            return y
        return channel_name

    def _ytmusic_keep_english_artist_for_search(
        self, channel_display_name: str, ytm_author: str
    ) -> bool:
        """Data API 為純英文且頻道類型像西方合輯，YTM 卻給中文在地名時，保留英文以利搜尋。"""
        if not channel_display_name or not ytm_author:
            return False
        ytm_cjk = bool(re.search(r"[\u4e00-\u9fff]", ytm_author))
        ch_cjk = bool(re.search(r"[\u4e00-\u9fff]", channel_display_name))
        if not ytm_cjk or ch_cjk:
            return False
        return bool(_YTM_KEEP_ENGLISH_ARTIST_RE.search(channel_display_name))

    async def _enrich_track_artist_from_ytmusic_player(
        self, track: Track, video_info: dict | None = None
    ) -> None:
        """以 ytmusicapi get_song(videoDetails.author) 補齊在地化顯示名（策略見 ytmusic_enrich_artist）。"""
        if not self._ytmusic or not track.artists:
            return
        mode = (getattr(self, "_ytmusic_enrich_artist", None) or "always").strip().lower()
        if mode in ("off", "false", "0", "never", "no"):
            return
        title = (video_info or {}).get("title") or getattr(track, "name", None) or ""
        if mode in ("title_cjk", "smart", "if_title_cjk"):
            if not re.search(r"[\u4e00-\u9fff]", title):
                return
        vid = track.item_id
        # author 快取：同步/重整時 MA 會對大量 track 重複呼叫 get_track，
        # 避免每首歌都多打一次 YTM get_song
        cached = self._ytm_author_cache.get(vid)
        if cached and time.time() < cached[0]:
            author = cached[1]
        else:
            loop = asyncio.get_event_loop()
            try:
                data = await loop.run_in_executor(
                    None, lambda v=vid: self._ytmusic.get_song(v),
                )
            except Exception as exc:
                self.logger.debug(f"[YouTube] ytmusic get_song 補藝人名失敗: {exc}")
                return
            vd = data.get("videoDetails") or {}
            author = (vd.get("author") or "").strip()
            if len(self._ytm_author_cache) > 2000:
                now_ts = time.time()
                for key in [
                    k for k, v in self._ytm_author_cache.items() if v[0] <= now_ts
                ]:
                    del self._ytm_author_cache[key]
            self._ytm_author_cache[vid] = (time.time() + 24 * 3600, author)
        if not author:
            return
        a0 = track.artists[0]
        old = getattr(a0, "name", "") or ""
        if self._ytmusic_keep_english_artist_for_search(old, author):
            self.logger.debug(
                f"[YouTube] 藝人名保留英文（略過 YTM {author!r}）: {old!r}"
            )
            return
        new = self._prefer_ytmusic_author_name(author, old)
        if new != old and new:
            a0.name = new
            self.logger.debug(f"[YouTube] 藝人名 YTM 補齊: {old!r} → {new!r}")

    @staticmethod
    def _parse_title(title: str, channel_name: str = "") -> tuple[str, str]:
        """從影片標題解析 (track_name, artist_name)。

        解析策略（依優先順序）：
        1. Artist《Title》「Title」『Title』→ 高把握，直接解析
        2. 左邊是純中文人名（2-5字）+ 分隔符 → artist=左, title=右
        3. 頻道名與標題 left/right 之一吻合 → 頻道名那側是 artist
        4. 其他 A - B 格式 → title=右邊，artist 留空（由 _to_track 用頻道名）
        5. 無分隔符 → 清理雜訊後回傳 title，artist 留空

        回傳 (track_name, artist_name)。
        """
        if channel_name:
            channel_name = YouTubeProvider._normalize_youtube_topic_channel_name(
                channel_name
            )
        import re

        # 移除前綴雜訊：【MV】【官方】[Official] 等
        cleaned = re.sub(
            r"^(?:【[^】]*】|〔[^〕]*〕|\[[^\]]*\])\s*",
            "", title, flags=re.UNICODE
        ).strip()

        # 移除結尾的中文方括號版本說明（可連續多個，例如 MV【官方版】【HD】）
        cleaned = re.sub(
            r"(?:\s*(?:MV|HD|4K|【[^】]*】))+\s*$",
            "", cleaned, flags=re.IGNORECASE | re.UNICODE
        ).strip()
        # 移除後綴雜訊
        cleaned = re.sub(
            r"\s*\b(?:official\s*(?:mv|music\s*video|video|audio|lyrics?\s*video)?"
            r"|mv|music\s*video|lyrics?\s*video|hd|4k|official"
            r"|完整版|正式版|首播版|現場版|live|cover|翻唱)\b\s*$",
            "", cleaned, flags=re.IGNORECASE | re.UNICODE
        ).strip()
        # 移除結尾括號內的版本/發行說明（保留 feat/ft）
        cleaned = re.sub(
            r"\s*[（(](?!.*\b(?:feat|ft)\.?\b)[^)）]*"
            r"(?:official|版|ver\.?|version|remix|edit|remaster)[^)）]*[)）]\s*$",
            "", cleaned, flags=re.IGNORECASE | re.UNICODE
        ).strip()

        # 格式1：Artist《Title》「Title」『Title』→ 高把握
        m = re.match(r"^(.+?)\s*[《「『](.+?)[》」』]", cleaned)
        if m:
            return m.group(2).strip(), m.group(1).strip()

        # 頻道名清理輔助（移除常見後綴和括號，用於比對）
        def _clean_ch(ch: str) -> str:
            ch = re.sub(
                r"\s*(?:official|music|vevo|records?|entertainment|topic)\s*$",
                "", ch, flags=re.IGNORECASE
            ).strip()
            ch = re.sub(r"\s*[\(\（][^\)\）]*[\)\）]", "", ch).strip()
            return ch.lower()

        ch_clean = _clean_ch(channel_name) if channel_name else ""

        # cjk_only_seps：這些分隔符只在左邊是純中文人名（格式2）時才採用，
        # 避免如 '五★一 - 某歌' 這類名稱中含有 ★ 的情況誤判
        cjk_only_seps = {"★", "☆", "◆", "●"}
        separators = [" - ", " – ", " — ", "－", "—", "★", "☆", "◆", "●"]
        for sep in separators:
            if sep in cleaned:
                left, right = cleaned.split(sep, 1)
                left, right = left.strip(), right.strip()
                if not left or not right:
                    continue

                # 格式2：左邊是純中文人名（2-5字）
                if re.match(r"^[\u4e00-\u9fff·•]{2,5}$", left):
                    return right, left

                # 特殊符號（★☆◆●）只在格式2 時採用，否則跳過繼續嘗試下一個分隔符
                if sep in cjk_only_seps:
                    continue

                # 格式3：頻道名比對（移除括號再比對）
                if ch_clean:
                    left_clean  = re.sub(r"\s*[\(\（][^\)\）]*[\)\）]", "", left).strip().lower()
                    right_clean = re.sub(r"\s*[\(\（][^\)\）]*[\)\）]", "", right).strip().lower()
                    if ch_clean in left_clean or left_clean in ch_clean:
                        return right, left   # left 是 artist
                    if ch_clean in right_clean or right_clean in ch_clean:
                        return left, right   # right 是 artist

                # 格式4：無法判斷，title 取右邊，artist 留空
                return right, ""

        return cleaned, ""

    def _to_track(self, data: dict) -> Track | None:
        video_id = data.get("id")
        if not video_id:
            return None

        # 過濾 yt-dlp flat-playlist 回傳的無法存取影片
        raw_title = data.get("title", "")
        if raw_title in ("[Private video]", "[Deleted video]", "Private video", "Deleted video"):
            self.logger.debug(f"[YouTube] 跳過無法存取影片: {video_id} ({raw_title})")
            return None
        if not raw_title:
            raw_title = "Unknown Title"

        track_mapping = ProviderMapping(
            item_id=video_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
        )

        channel_id   = data.get("channel_id") or data.get("uploader_id") or "youtube_unknown"
        channel_name = self._normalize_youtube_topic_channel_name(
            data.get("channel") or data.get("uploader") or "YouTube"
        )
        # raw_title 已在上方設定並驗證

        # 嘗試從標題解析歌名和 artist（傳入頻道名輔助判斷）
        parsed_title, parsed_artist = self._parse_title(raw_title, channel_name)

        # Artist 決策：
        # - parsed_artist 是純中文人名（2-5字）→ 高把握，直接採用（如 '华晨宇★...' 中的 '华晨宇'）
        # - 頻道名是通用名且標題有解析出 artist → 用標題解析的
        # - 其他 → 用頻道名（最穩定，適合頻道即 artist 的情況）
        import re as _re
        is_generic_channel = channel_name.lower().strip() in self._GENERIC_CHANNEL_NAMES
        _is_cjk_name = bool(parsed_artist and _re.match(r'^[\u4e00-\u9fff·•]{2,5}$', parsed_artist))
        if _is_cjk_name:
            display_artist = parsed_artist  # 標題解析出純中文人名，優先採用
        elif parsed_artist and is_generic_channel:
            display_artist = parsed_artist
        else:
            display_artist = channel_name

        # 歌名：有解析出來就用解析的，否則用原始標題
        track_name = parsed_title if parsed_title else raw_title
        self.logger.debug(
            f"[YouTube] _to_track: raw={raw_title!r} → name={track_name!r} artist={display_artist!r}"
        )

        artist_mapping = ProviderMapping(
            item_id=channel_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/channel/{channel_id}",
        )
        artist = Artist(
            item_id=channel_id,
            name=display_artist,
            provider=self.instance_id,
            provider_mappings={artist_mapping},
        )

        track = Track(
            item_id=video_id,
            name=track_name,
            artists=[artist],
            provider=self.instance_id,
            duration=int(data.get("duration") or 0),
            provider_mappings={track_mapping},
        )

        thumbnails = data.get("thumbnails")
        thumbnail  = data.get("thumbnail")
        thumb_url  = thumbnails[-1]["url"] if thumbnails else thumbnail
        if thumb_url:
            track.metadata.images = [
                MediaItemImage(type=ImageType.THUMB, path=thumb_url, provider=self.instance_id)
            ]
        return track

    def _channel_to_artist(
        self, channel_id: str, channel_name: str, data: dict
    ) -> Artist | None:
        """將頻道資訊轉為 MA Artist 物件。"""
        channel_name = self._normalize_youtube_topic_channel_name(channel_name)
        mapping = ProviderMapping(
            item_id=channel_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/channel/{channel_id}",
        )
        artist = Artist(
            item_id=channel_id,
            name=channel_name,
            provider=self.instance_id,
            provider_mappings={mapping},
        )
        # 頻道縮圖
        thumbnails = data.get("thumbnails")
        thumbnail  = data.get("thumbnail")
        thumb_url  = thumbnails[-1]["url"] if thumbnails else thumbnail
        if thumb_url:
            artist.metadata.images = [
                MediaItemImage(type=ImageType.THUMB, path=thumb_url, provider=self.instance_id)
            ]
        return artist

    async def _get_artist_via_ytmusic(self, browse_id: str) -> Artist:
        """透過 ytmusicapi get_artist 取得 YouTube Music 藝人頁。

        YTMusic 藝人使用 musicImmersiveHeaderRenderer；若只用 InnerTube（TV）
        走 www.youtube.com 頻道頁的 c4Tabbed／pageHeader，常得到空名稱與縮圖。
        """
        if not self._ytmusic:
            raise RuntimeError("ytmusicapi 未初始化")
        bid = self._ytmusic_normalize_artist_browse_id(browse_id)
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda b=bid: self._ytmusic.get_artist(b),
        )
        name = (data.get("name") or "").strip()
        if not name:
            raise RuntimeError(f"ytmusic get_artist 無有效 name: {browse_id!r}")
        thumbs = data.get("thumbnails") or []
        thumb_url = self._ytmusic_best_thumb(thumbs)
        mapping = ProviderMapping(
            item_id=browse_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://music.youtube.com/channel/{bid}",
        )
        artist = Artist(
            item_id=browse_id,
            name=name,
            provider=self.instance_id,
            provider_mappings={mapping},
        )
        if thumb_url:
            artist.metadata.images = [
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=thumb_url,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )
            ]
        self.logger.debug(
            f"[YouTube] _get_artist_via_ytmusic: {browse_id!r} name={name!r}"
        )
        return artist

    async def get_artist(self, item_id: str) -> Artist:
        """取得頻道／藝人資訊。ytmusicapi（若有）→ InnerTube → yt-dlp。"""
        await self._ensure_fresh_token()
        methods: list = []
        if self._ytmusic:
            methods.append(("ytmusic", self._get_artist_via_ytmusic))
        methods.extend(
            [
                ("InnerTube", self._get_artist_via_innertube),
                ("yt-dlp", self._get_artist_via_ytdlp),
            ]
        )
        return await self._try_methods("取得 artist", methods, item_id)

    async def _get_artist_via_api(self, channel_id: str) -> Artist:
        """透過 YouTube Data API v3 取得頻道資訊。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {"part": "snippet,statistics", "id": channel_id}
        async with self._session.get(
            f"{YOUTUBE_API_BASE}/channels", headers=headers, params=params,
        ) as resp:
            if resp.status == 401:
                await self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/channels", headers=headers, params=params,
                ) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()

        items = data.get("items", [])
        if not items:
            raise RuntimeError(f"找不到頻道: {channel_id}")

        snippet      = items[0].get("snippet", {})
        channel_name = snippet.get("title", channel_id)
        thumbnails   = snippet.get("thumbnails", {})

        mapping = ProviderMapping(
            item_id=channel_id, provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/channel/{channel_id}",
        )
        artist = Artist(
            item_id=channel_id, name=channel_name,
            provider=self.instance_id, provider_mappings={mapping},
        )
        for res in ("high", "medium", "default"):
            if res in thumbnails:
                artist.metadata.images = [MediaItemImage(
                    type=ImageType.THUMB, path=thumbnails[res]["url"],
                    provider=self.instance_id,
                )]
                break
        return artist

    async def _get_artist_via_ytdlp(self, item_id: str) -> Artist:
        """透過 yt-dlp 取得頻道資訊（fallback）。"""
        url  = f"https://www.youtube.com/channel/{item_id}"
        args = (
            ["yt-dlp", "--quiet", "--no-warnings", "--dump-json",
             "--playlist-items", "1", "--flat-playlist"]
            + self._bgutil_args()
            + self._auth_args()
            + [url]
        )
        channel_name = None
        thumb_url    = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if stdout.strip():
                for line in stdout.decode().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        channel_name = (
                            d.get("channel") or d.get("uploader")
                            or d.get("playlist_uploader") or d.get("channel_id")
                        )
                        thumbnails = d.get("thumbnails")
                        thumb_url  = thumbnails[-1]["url"] if thumbnails else d.get("thumbnail")
                        break
                    except Exception:
                        continue
        except Exception as exc:
            self.logger.warning(f"[YouTube] _get_artist_via_ytdlp {item_id} 例外: {exc}")

        mapping = ProviderMapping(
            item_id=item_id, provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/channel/{item_id}",
        )
        disp_name = self._normalize_youtube_topic_channel_name(channel_name or item_id)
        artist = Artist(
            item_id=item_id, name=disp_name,
            provider=self.instance_id, provider_mappings={mapping},
        )
        if thumb_url:
            artist.metadata.images = [
                MediaItemImage(type=ImageType.THUMB, path=thumb_url, provider=self.instance_id)
            ]
        return artist

    async def _get_artist_toptracks_via_ytmusic(self, prov_artist_id: str) -> list[Track]:
        """透過 ytmusicapi 藝人頁的 songs／videos 清單（get_playlist）取得主打曲目。"""
        if not self._ytmusic:
            raise RuntimeError("ytmusicapi 未初始化")
        bid = self._ytmusic_normalize_artist_browse_id(prov_artist_id)
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: self._ytmusic.get_artist(bid),
        )
        tracks: list[Track] = []

        def _append_from_playlist(pl_browse_id: str) -> None:
            pl = self._ytmusic.get_playlist(pl_browse_id, limit=100)
            for item in pl.get("tracks") or []:
                try:
                    tracks.append(self._ytmusic_track_to_ma(item))
                except Exception as exc:
                    self.logger.debug(
                        f"[YouTube] ytmusic 藝人清單 track 轉換失敗: {exc}"
                    )

        def _append_from_results(items: list) -> None:
            for item in items:
                if not item.get("videoId"):
                    continue
                try:
                    tracks.append(self._ytmusic_track_to_ma(item))
                except Exception as exc:
                    self.logger.debug(
                        f"[YouTube] ytmusic 藝人 shelf track 轉換失敗: {exc}"
                    )

        for section_key in ("songs", "videos"):
            section = data.get(section_key) or {}
            pl_browse = section.get("browseId")
            if pl_browse:
                try:
                    await loop.run_in_executor(
                        None, lambda b=pl_browse: _append_from_playlist(b),
                    )
                except Exception as exc:
                    self.logger.debug(
                        f"[YouTube] get_playlist({section_key}) 失敗: {exc}"
                    )
            if not tracks:
                _append_from_results(section.get("results") or [])
            if tracks:
                break

        seen: set[str] = set()
        unique: list[Track] = []
        for t in tracks:
            if t.item_id in seen:
                continue
            seen.add(t.item_id)
            unique.append(t)

        self.logger.debug(
            f"[YouTube] _get_artist_toptracks_via_ytmusic {prov_artist_id}: "
            f"{len(unique)} 首"
        )
        return unique

    async def get_artist_toptracks(self, prov_artist_id: str) -> list[Track]:
        """列出藝人主打曲目。ytmusicapi（若有）→ InnerTube 頻道影片 → yt-dlp。"""
        self.logger.info(
            f"[YouTube] get_artist_toptracks called: {prov_artist_id}, "
            f"features={[f.value for f in self._supported_features]}"
        )
        await self._ensure_fresh_token()

        methods: list = []
        if self._ytmusic:
            methods.append(("ytmusic", self._get_artist_toptracks_via_ytmusic))
        methods.extend(
            [
                ("InnerTube", self._get_channel_videos_via_innertube),
                ("yt-dlp", self._get_channel_videos_via_ytdlp),
            ]
        )
        return await self._try_methods("取得頻道影片", methods, prov_artist_id)

    async def _get_channel_videos_via_api(self, channel_id: str) -> list[Track]:
        """用 YouTube Data API v3 取得頻道影片。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {
            "part": "snippet", "channelId": channel_id, "type": "video",
            "order": "date", "maxResults": "50",
        }
        tracks: list[Track] = []
        async with self._session.get(
            f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
        ) as resp:
            if resp.status == 401:
                await self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/search", headers=headers, params=params,
                ) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()
        for item in data.get("items", []):
            vid_id  = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not vid_id:
                continue
            channel_name = snippet.get("channelTitle", "")
            artist_mapping = ProviderMapping(
                item_id=channel_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
                url=f"https://www.youtube.com/channel/{channel_id}",
            )
            raw_title = snippet.get("title", "")
            parsed_title, parsed_artist = self._parse_title(raw_title, channel_name)
            track_name  = parsed_title or raw_title
            artist_name = parsed_artist or self._normalize_youtube_topic_channel_name(
                channel_name
            )
            artist = Artist(item_id=channel_id, name=artist_name,
                            provider=self.instance_id, provider_mappings={artist_mapping})
            track_mapping = ProviderMapping(
                item_id=vid_id, provider_domain=self.domain, provider_instance=self.instance_id,
            )
            track = Track(item_id=vid_id, name=track_name,
                          artists=[artist], provider=self.instance_id,
                          provider_mappings={track_mapping})
            thumbnails = snippet.get("thumbnails", {})
            for res in ("high", "medium", "default"):
                if res in thumbnails:
                    track.metadata.images = [MediaItemImage(
                        type=ImageType.THUMB, path=thumbnails[res]["url"],
                        provider=self.instance_id,
                    )]
                    break
            tracks.append(track)
        # 批次補上 duration
        if tracks and self._access_token:
            durations = await self._fetch_video_durations([t.item_id for t in tracks])
            for track in tracks:
                if track.item_id in durations:
                    track.duration = durations[track.item_id]

        self.logger.debug(f"[YouTube] Data API 取得頻道 {channel_id} 共 {len(tracks)} 支影片")
        return tracks

    async def _get_channel_videos_via_ytdlp(self, channel_id: str) -> list[Track]:
        """用 yt-dlp 取得頻道影片（fallback）。"""
        url = f"https://www.youtube.com/channel/{channel_id}/videos"
        args = (
            ["yt-dlp", "--quiet", "--no-warnings",
             "--flat-playlist", "--dump-json", "--playlist-end", "100"]
            + self._bgutil_args()
            + self._auth_args()
            + [url]
        )
        tracks: list[Track] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr_data = await proc.communicate()
            if proc.returncode != 0 and not stdout:
                self.logger.error(
                    f"[YouTube] get_artist_toptracks 失敗: {stderr_data.decode().strip()}"
                )
                return tracks
            for line in stdout.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not data.get("channel_id"):
                        data["channel_id"] = channel_id
                    track = self._to_track(data)
                    if track:
                        tracks.append(track)
                except Exception as exc:
                    self.logger.debug(f"[YouTube] 解析 track 失敗: {exc}")
        except Exception as exc:
            self.logger.error(f"[YouTube] _get_channel_videos_via_ytdlp 例外: {exc}", exc_info=True)
        self.logger.debug(f"[YouTube] yt-dlp 取得頻道 {channel_id} 共 {len(tracks)} 支影片")
        return tracks

    @staticmethod
    def _ytmusic_looks_like_year_field_not_artist(s: str) -> bool:
        """ytmusicapi 有時把年份誤塞進 artists（如 id/name 為 2024年）。"""
        if not s or not isinstance(s, str):
            return False
        t = s.strip()
        if re.match(r"^(\d{4})年?\s*$", t):
            return True
        if re.match(r"^\d{4}\s*$", t):
            return True
        return False

    def _ytmusic_album_artist_entry_usable(self, a: dict) -> bool:
        if not isinstance(a, dict):
            return False
        aid = (a.get("id") or "").strip()
        name = (a.get("name") or "").strip()
        if self._ytmusic_looks_like_year_field_not_artist(name):
            return False
        if self._ytmusic_looks_like_year_field_not_artist(aid):
            return False
        return bool(name or aid)

    def _ytmusic_artist_dict_to_album_item_mapping(self, a: dict) -> ItemMapping | None:
        """專輯上的藝人用 ItemMapping（與官方 ytmusic provider 一致），避免 SearchResults 序列化失敗。"""
        if not self._ytmusic_album_artist_entry_usable(a):
            return None
        aid = (a.get("id") or "").strip() or (a.get("name") or "").strip()
        if not aid:
            return None
        aname = (a.get("name") or "").strip() or aid
        return ItemMapping(
            item_id=aid,
            name=aname,
            provider=self.instance_id,
            media_type=MediaType.ARTIST,
        )

    def _ytmusic_catalog_album_to_ma(
        self,
        album: dict,
        *,
        page_artist_browse_id: str = "",
        page_artist_name: str = "",
    ) -> Album | None:
        """將 get_artist_albums／get_artist 的專輯條目轉為 MA Album（browseId 多為 MPREb_）。"""
        browse_id = album.get("browseId")
        if not browse_id:
            return None
        title = album.get("title", browse_id)
        thumb_url = self._ytmusic_best_thumb(album.get("thumbnails") or [])
        mapping = ProviderMapping(
            item_id=browse_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://music.youtube.com/browse/{browse_id}",
        )
        out = Album(
            item_id=browse_id,
            provider=self.instance_id,
            name=title,
            provider_mappings={mapping},
        )
        year = album.get("year")
        if year:
            ys = str(year).strip()
            try:
                if len(ys) >= 4 and ys[:4].isdigit():
                    out.year = int(ys[:4])
            except (ValueError, TypeError):
                pass
        artist_list: list[ItemMapping] = []
        for a in album.get("artists") or []:
            if im := self._ytmusic_artist_dict_to_album_item_mapping(a):
                artist_list.append(im)
        if not artist_list and page_artist_browse_id:
            bid = page_artist_browse_id.strip()
            nm = (page_artist_name or "").strip() or bid
            artist_list.append(
                ItemMapping(
                    item_id=bid,
                    name=nm,
                    provider=self.instance_id,
                    media_type=MediaType.ARTIST,
                )
            )
        if artist_list:
            out.artists = UniqueList(artist_list)
        if thumb_url:
            out.metadata.images = [
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=thumb_url,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )
            ]
        return out

    async def _get_artist_albums_via_ytmusic(self, prov_artist_id: str) -> list[Album]:
        """透過 ytmusicapi get_artist + get_artist_albums 取得專輯與單曲。"""
        if not self._ytmusic:
            raise RuntimeError("ytmusicapi 未初始化")
        bid = self._ytmusic_normalize_artist_browse_id(prov_artist_id)
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: self._ytmusic.get_artist(bid),
        )
        page_artist_name = (data.get("name") or "").strip()
        albums_out: list[Album] = []

        for section_key in ("albums", "singles"):
            section = data.get(section_key) or {}
            browse_id = section.get("browseId")
            params = section.get("params")
            results = section.get("results") or []
            if browse_id and params:
                try:
                    full = await loop.run_in_executor(
                        None,
                        lambda b=browse_id, p=params: self._ytmusic.get_artist_albums(
                            b, p, limit=100
                        ),
                    )
                    for album in full:
                        ma = self._ytmusic_catalog_album_to_ma(
                            album,
                            page_artist_browse_id=bid,
                            page_artist_name=page_artist_name,
                        )
                        if ma:
                            albums_out.append(ma)
                except Exception as exc:
                    self.logger.debug(
                        f"[YouTube] get_artist_albums({section_key}) 失敗: {exc}"
                    )
                    for album in results:
                        ma = self._ytmusic_catalog_album_to_ma(
                            album,
                            page_artist_browse_id=bid,
                            page_artist_name=page_artist_name,
                        )
                        if ma:
                            albums_out.append(ma)
            else:
                for album in results:
                    ma = self._ytmusic_catalog_album_to_ma(
                        album,
                        page_artist_browse_id=bid,
                        page_artist_name=page_artist_name,
                    )
                    if ma:
                        albums_out.append(ma)

        self.logger.debug(
            f"[YouTube] _get_artist_albums_via_ytmusic {prov_artist_id}: "
            f"{len(albums_out)} 張"
        )
        return albums_out

    async def _get_artist_albums_via_channel_playlists(
        self, prov_artist_id: str
    ) -> list[Album]:
        """以頻道公開播放清單模擬專輯（舊行為）。"""
        playlists = await self._get_channel_playlists(prov_artist_id)
        albums: list[Album] = []
        for pl in playlists:
            album = self._playlist_to_album(pl, prov_artist_id)
            if album:
                albums.append(album)
        self.logger.debug(
            f"[YouTube] 頻道 {prov_artist_id} 共 {len(albums)} 個播放清單（專輯）"
        )
        return albums

    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """取得藝人專輯／單曲。ytmusicapi（若有）→ 頻道播放清單。"""
        await self._ensure_fresh_token()
        methods: list = []
        if self._ytmusic:
            methods.append(("ytmusic", self._get_artist_albums_via_ytmusic))
        methods.append(("頻道 playlists", self._get_artist_albums_via_channel_playlists))
        return await self._try_methods("取得 artist albums", methods, prov_artist_id)

    async def _ytdlp_playlist_meta(self, playlist_id: str) -> dict:
        """用 yt-dlp 取得播放清單 metadata（標題、縮圖），不下載影片。"""
        url  = f"https://www.youtube.com/playlist?list={playlist_id}"
        args = (
            ["yt-dlp", "--quiet", "--no-warnings",
             "--flat-playlist", "--dump-single-json",
             "--playlist-items", "0"]
            + self._bgutil_args()
            + self._auth_args()
            + [url]
        )
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if not stdout:
            raise RuntimeError(f"yt-dlp 取不到播放清單 metadata: {playlist_id}")
        return json.loads(stdout.decode())

    async def get_album(self, prov_album_id: str) -> Album:
        """取得單一專輯資訊。MPREb_ 開頭的用 ytmusicapi，其餘用 yt-dlp。"""
        await self._ensure_fresh_token()

        if prov_album_id.startswith("MPREb_") and self._ytmusic:
            return await self._get_album_via_ytmusic(prov_album_id)

        try:
            data  = await self._ytdlp_playlist_meta(prov_album_id)
            title = data.get("title") or data.get("playlist_title") or prov_album_id
            thumbnails = data.get("thumbnails")
            thumbnail  = data.get("thumbnail")
            thumb = thumbnails[-1]["url"] if thumbnails else thumbnail or ""
            channel_id   = data.get("channel_id") or data.get("uploader_id") or ""
            channel_name = data.get("channel") or data.get("uploader") or ""
            self.logger.debug(f"[YouTube] get_album: {prov_album_id} title={title!r} channel={channel_name!r}")
        except Exception as exc:
            self.logger.debug(f"[YouTube] get_album yt-dlp 失敗: {exc}")
            title        = prov_album_id
            thumb        = ""
            channel_id   = ""
            channel_name = ""

        mapping = ProviderMapping(
            item_id=prov_album_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/playlist?list={prov_album_id}",
        )
        album = Album(
            item_id=prov_album_id,
            provider=self.instance_id,
            name=title,
            provider_mappings={mapping},
        )
        if channel_id and channel_name:
            artist_mapping = ProviderMapping(
                item_id=channel_id,
                provider_domain=self.domain,
                provider_instance=self.instance_id,
                url=f"https://www.youtube.com/channel/{channel_id}",
            )
            album.artists = [Artist(
                item_id=channel_id,
                name=channel_name,
                provider=self.instance_id,
                provider_mappings={artist_mapping},
            )]
        if thumb:
            album.metadata.images = [
                MediaItemImage(type=ImageType.THUMB, path=thumb, provider=self.instance_id, remotely_accessible=True)
            ]
        return album

    async def _get_album_via_ytmusic(self, browse_id: str) -> Album:
        """透過 ytmusicapi 取得 YouTube Music 專輯資訊（MPREb_ browseId）。"""
        data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._ytmusic.get_album(browse_id),
        )
        title = data.get("title", browse_id)
        year = data.get("year")
        thumbs = data.get("thumbnails") or []
        thumb_url = self._ytmusic_best_thumb(thumbs)

        mapping = ProviderMapping(
            item_id=browse_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://music.youtube.com/browse/{browse_id}",
        )
        album = Album(
            item_id=browse_id,
            provider=self.instance_id,
            name=title,
            provider_mappings={mapping},
        )
        if year:
            try:
                album.year = int(year)
            except (ValueError, TypeError):
                pass

        artist_list: list[ItemMapping] = []
        for a in data.get("artists") or []:
            if im := self._ytmusic_artist_dict_to_album_item_mapping(a):
                artist_list.append(im)
        if artist_list:
            album.artists = UniqueList(artist_list)

        if thumb_url:
            album.metadata.images = [
                MediaItemImage(type=ImageType.THUMB, path=thumb_url, provider=self.instance_id, remotely_accessible=True)
            ]
        self.logger.debug(f"[YouTube] get_album (ytmusicapi): {browse_id} → {title!r}")
        return album

    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """取得「專輯」的所有曲目。MPREb_ 用 ytmusicapi，其餘用 InnerTube。"""
        if prov_album_id.startswith("MPREb_") and self._ytmusic:
            return await self._get_album_tracks_via_ytmusic(prov_album_id)
        return await self._get_playlist_tracks_via_innertube(prov_album_id)

    async def _get_album_tracks_via_ytmusic(self, browse_id: str) -> list[Track]:
        """透過 ytmusicapi 取得 YouTube Music 專輯的曲目列表。"""
        data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._ytmusic.get_album(browse_id),
        )
        album_name = data.get("title", browse_id)
        tracks: list[Track] = []
        for idx, item in enumerate(data.get("tracks") or [], start=1):
            vid = item.get("videoId")
            if not vid:
                continue
            try:
                track = self._ytmusic_track_to_ma(item)
                track.disc_number = 1
                track.track_number = idx
                if not track.album and album_name:
                    track.album = ItemMapping(
                        media_type=MediaType.ALBUM,
                        item_id=browse_id,
                        provider=self.instance_id,
                        name=album_name,
                    )
                tracks.append(track)
            except Exception as exc:
                self.logger.debug(f"[YouTube] 轉換 album track 失敗: {exc}")
        self.logger.debug(f"[YouTube] get_album_tracks (ytmusicapi): {browse_id} → {len(tracks)} 首")
        return tracks

    async def _get_channel_playlists(self, channel_id: str) -> list[dict]:
        """取得頻道的播放清單列表。Data API v3（custom 模式）→ InnerTube → yt-dlp。"""
        if self._is_api_available():
            try:
                return await self._get_channel_playlists_via_api(channel_id)
            except Exception as exc:
                self.logger.debug(f"[YouTube] Data API 取頻道 playlists 失敗: {exc}")
        playlists = await self._get_channel_playlists_via_innertube(channel_id)
        if playlists:
            return playlists
        return await self._get_channel_playlists_via_ytdlp(channel_id)

    async def _get_channel_playlists_via_api(self, channel_id: str) -> list[dict]:
        """透過 YouTube Data API v3 取得頻道播放清單。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {
            "part":       "snippet",
            "channelId":  channel_id,
            "maxResults": "50",
        }
        playlists = []
        while True:
            async with self._session.get(
                f"{YOUTUBE_API_BASE}/playlists", headers=headers, params=params
            ) as resp:
                if resp.status == 401:
                    await self._refresh_access_token()
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    continue
                resp.raise_for_status()
                data = await resp.json()
                for item in data.get("items", []):
                    pl_id = item.get("id")
                    snippet = item.get("snippet", {})
                    if not pl_id:
                        continue
                    playlists.append({
                        "id":        pl_id,
                        "title":     snippet.get("title", ""),
                        "thumbnail": self._best_thumbnail(snippet.get("thumbnails", {})),
                    })
                next_page = data.get("nextPageToken")
                if not next_page or len(playlists) >= 200:
                    break
                params["pageToken"] = next_page
        return playlists

    async def _get_channel_playlists_via_innertube(self, channel_id: str) -> list[dict]:
        """透過 InnerTube 取得頻道播放清單。"""
        try:
            data  = await self._innertube_request(
                "browse", {"browseId": channel_id, "params": "EglwbGF5bGlzdHMQAQ=="},
                use_web_client=True,
            )
        except Exception as exc:
            self.logger.debug(f"[YouTube] InnerTube 取頻道 playlists 失敗: {exc}")
            return []
        playlists = []
        grid_items = self._innertube_find_all(data, "gridPlaylistRenderer")
        list_items = self._innertube_find_all(data, "playlistRenderer")
        self.logger.debug(
            f"[YouTube] _get_channel_playlists_via_innertube {channel_id}: "
            f"gridPlaylistRenderer={len(grid_items)} playlistRenderer={len(list_items)}"
        )
        items = grid_items + list_items
        for item in items:
            pl_id = item.get("playlistId")
            if not pl_id:
                continue
            title  = self._innertube_extract_text(item.get("title", {}))
            thumbs = item.get("thumbnail", {}).get("thumbnails", [])
            thumb  = thumbs[-1]["url"] if thumbs else ""
            playlists.append({"id": pl_id, "title": title, "thumbnail": thumb})
        return playlists

    async def _get_channel_playlists_via_ytdlp(self, channel_id: str) -> list[dict]:
        """透過 yt-dlp 取得頻道播放清單（InnerTube fallback）。"""
        url  = f"https://www.youtube.com/channel/{channel_id}/playlists"
        args = (
            ["yt-dlp", "--quiet", "--no-warnings", "--flat-playlist", "--dump-json"]
            + self._auth_args()
            + [url]
        )
        self.logger.debug(f"[YouTube] yt-dlp 取頻道 {channel_id} 播放清單…")
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
        except Exception as exc:
            self.logger.debug(f"[YouTube] yt-dlp 取頻道 playlists 例外: {exc}")
            return []

        playlists = []
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pl_id = data.get("id") or data.get("playlist_id")
                title = data.get("title") or data.get("playlist_title", "")
                thumb = ""
                thumbs = data.get("thumbnails") or []
                if thumbs:
                    thumb = thumbs[-1].get("url", "")
                if pl_id and title:
                    playlists.append({"id": pl_id, "title": title, "thumbnail": thumb})
            except Exception:
                continue
        self.logger.debug(f"[YouTube] yt-dlp 取頻道 {channel_id} 共 {len(playlists)} 個播放清單")
        return playlists

    def _playlist_to_album(self, pl: dict, artist_id: str) -> Album | None:
        """將頻道播放清單轉為 MA Album 物件。"""
        pl_id = pl.get("id")
        title = pl.get("title", "")
        if not pl_id or not title:
            return None
        mapping = ProviderMapping(
            item_id=pl_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/playlist?list={pl_id}",
        )
        artist_mapping = ProviderMapping(
            item_id=artist_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
        )
        artist = Artist(
            item_id=artist_id,
            provider=self.instance_id,
            name="",  # MA 會從 library 補上
            provider_mappings={artist_mapping},
        )
        album = Album(
            item_id=pl_id,
            provider=self.instance_id,
            name=title,
            artists=[artist],
            provider_mappings={mapping},
        )
        thumb = pl.get("thumbnail", "")
        if thumb:
            album.metadata.images = [
                MediaItemImage(type=ImageType.THUMB, path=thumb, provider=self.instance_id)
            ]
        return album

    @staticmethod
    def _best_thumbnail(thumbnails: dict) -> str:
        """從 Data API 縮圖 dict 取最高解析度 URL。"""
        for res in ("maxres", "high", "medium", "default"):
            if res in thumbnails:
                return thumbnails[res].get("url", "")
        return ""

    async def get_track(self, item_id: str) -> Track:
        try:
            info = await self._video_info(item_id)
        except RuntimeError as exc:
            raise MediaNotFoundError(f"YouTube 影片無法取得: {item_id}") from exc
        track = self._to_track(info)
        if not track:
            raise MediaNotFoundError(f"YouTube 影片無法解析: {item_id}")
        await self._enrich_track_artist_from_ytmusic_player(track, info)
        return track

    # ------------------------------------------------------------------
    # Playlist 支援（需登入）
    # ------------------------------------------------------------------

    async def get_library_playlists(self) -> AsyncGenerator[Playlist, None]:
        """
        列出使用者帳號的所有播放清單。

        三種模式的策略：
          tv     → InnerTube TV client + Bearer（FEplaylist_aggregation）
          custom → Data API v3（mine=true）→ InnerTube fallback
          ytdlp  → yt-dlp + cookies
        共通：cookies 存在時合併 /feed/playlists（含已儲存他人清單）。
        """
        await self._ensure_fresh_token()

        if not self._access_token and not self._cookies_file:
            raise UnplayableMediaError(
                "取得個人播放清單需要 OAuth 登入（設定 → YouTube → 開始登入）"
            )

        seen: set[str] = set()
        any_yielded = False

        # 0. 觀看紀錄假清單（需要 cookies.txt）
        if self._cookies_file:
            history_mapping = ProviderMapping(
                item_id="__yt_history__",
                provider_domain=self.domain,
                provider_instance=self.instance_id,
                url="https://www.youtube.com/feed/history",
            )
            history_pl = Playlist(
                item_id="__yt_history__",
                name="▶ 觀看紀錄（最近 50 筆）",
                provider=self.instance_id,
                provider_mappings={history_mapping},
                is_editable=False,
            )
            seen.add("__yt_history__")
            yield history_pl
            any_yielded = True

        # 1. 依模式取得自建播放清單
        if self._auth_mode == "custom" and self._is_api_available():
            # custom 模式：Data API v3（mine=true）
            try:
                async for playlist in self._get_playlists_via_api():
                    if playlist.item_id not in seen:
                        seen.add(playlist.item_id)
                        yield playlist
                        any_yielded = True
                self.logger.debug(f"[YouTube] Data API v3 共取得 {len(seen)} 個自建清單")
            except Exception as exc:
                err_str = str(exc)
                if "403" in err_str or "quota" in err_str.lower():
                    self._handle_quota_exceeded()
                self.logger.warning(f"[YouTube] Data API v3 失敗: {exc}")
        elif self._access_token:
            # tv 模式（或 custom 配額耗盡）：InnerTube TV client + Bearer
            try:
                async for playlist in self._get_playlists_via_innertube_tv():
                    if playlist.item_id not in seen:
                        seen.add(playlist.item_id)
                        yield playlist
                        any_yielded = True
                self.logger.debug(
                    f"[YouTube] InnerTube TV browse 共取得 {len(seen)} 個清單"
                )
            except Exception as exc:
                self.logger.warning(f"[YouTube] InnerTube TV browse 失敗: {exc}")

        # 2. cookies → /feed/playlists（包含已儲存他人清單）
        if self._cookies_file:
            try:
                async for playlist in self._get_playlists_via_innertube():
                    if playlist.item_id not in seen:
                        seen.add(playlist.item_id)
                        yield playlist
                        any_yielded = True
                self.logger.debug(
                    f"[YouTube] /feed/playlists 合計（含已儲存他人清單）: {len(seen)} 個"
                )
            except Exception as exc:
                self.logger.warning(f"[YouTube] /feed/playlists 失敗: {exc}")

        if any_yielded:
            return

        # 3. yt-dlp + cookies/auth（最終 fallback）
        if self._cookies_file or self._access_token:
            try:
                async for playlist in self._get_playlists_via_ytdlp():
                    if playlist.item_id not in seen:
                        seen.add(playlist.item_id)
                        yield playlist
                return
            except Exception as exc:
                self.logger.warning(f"[YouTube] yt-dlp 播放清單 fallback 失敗: {exc}")

        if not seen:
            raise UnplayableMediaError(
                "無法取得個人播放清單，請確認已完成 OAuth 登入"
            )

    async def _get_playlists_via_api(self) -> AsyncGenerator[Playlist, None]:
        """透過 YouTube Data API v3 取得播放清單（支援 OAuth token）。

        先完整收集所有清單，再一次性 yield，確保中途失敗時不 yield 部分資料。
        """
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params = {
            "part": "snippet,contentDetails",
            "mine": "true",
            "maxResults": "50",
        }

        playlists: list[Playlist] = []
        refreshed = False
        self.logger.debug("[YouTube] 透過 Data API v3 取得播放清單…")

        # 完整收集後才 yield，中途失敗會 raise 讓外層走 fallback
        while True:
            async with self._session.get(
                f"{YOUTUBE_API_BASE}/playlists",
                headers=headers,
                params=params,
            ) as resp:
                if resp.status == 401 and not refreshed:
                    self.logger.warning("[YouTube] API token 過期，嘗試刷新…")
                    await self._refresh_access_token()
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    refreshed = True
                    continue
                resp.raise_for_status()  # 失敗直接 raise，讓外層走 fallback
                data = await resp.json()

                raw_items = data.get("items", [])
                self.logger.info(
                    f"[YouTube] Data API v3 這頁回傳 {len(raw_items)} 筆, "
                    f"pageToken={'有' if params.get('pageToken') else '無'}"
                )
                for item in raw_items:
                    pl_id = item.get("id", "")
                    playlist = self._api_item_to_playlist(item)
                    if not playlist:
                        self.logger.debug(f"[YouTube] 跳過（無法解析）: {pl_id}")
                    else:
                        playlists.append(playlist)
                        self.logger.debug(f"[YouTube] 加入播放清單: {playlist.name} ({pl_id})")

                next_page = data.get("nextPageToken")
                self.logger.info(
                    f"[YouTube] Data API v3 nextPageToken={'有，繼續取' if next_page else '無，結束'}"
                )
                if not next_page:
                    break
                params["pageToken"] = next_page

        self.logger.info(f"[YouTube] Data API v3 共取得 {len(playlists)} 個播放清單")
        for pl in playlists:
            yield pl

    async def _get_playlists_via_innertube_tv(self) -> AsyncGenerator[Playlist, None]:
        """透過 InnerTube TV client + Bearer token 取得個人播放清單。

        使用 FEplaylist_aggregation browseId，TV client 回應格式不同於 WEB，
        透過泛用 finder 尋找 playlistRenderer / gridPlaylistRenderer / tileRenderer。
        """
        if not self._access_token:
            raise RuntimeError("InnerTube TV browse 需要 OAuth token")

        data = await self._innertube_request(
            "browse", {"browseId": "FEplaylist_aggregation"}
        )

        items = (
            self._innertube_find_all(
                data,
                "playlistRenderer",
                "compactPlaylistRenderer",
                "gridPlaylistRenderer",
            )
        )

        # TV client 可能用 tileRenderer 包裝 playlist
        for tile in self._innertube_find_all(data, "tileRenderer"):
            on_select = tile.get("onSelectCommand", {})
            pl_ep = on_select.get("browseEndpoint", {})
            pl_id = pl_ep.get("browseId", "")
            if pl_id.startswith("VL"):
                header = tile.get("header", {}).get("tileHeaderRenderer", {})
                metadata = tile.get("metadata", {}).get("tileMetadataRenderer", {})
                title = (
                    self._innertube_extract_text(metadata.get("title", {}))
                    or self._innertube_extract_text(header.get("title", {}))
                )
                thumbs = (
                    header.get("thumbnail", {})
                    .get("thumbnails", [])
                )
                thumb_url = thumbs[-1]["url"] if thumbs else ""
                items.append({
                    "playlistId": pl_id[2:],
                    "title": {"simpleText": title},
                    "thumbnail": {"thumbnails": [{"url": thumb_url}] if thumb_url else []},
                })

        if not items:
            renderer_keys = self._innertube_collect_renderer_keys(data)
            self.logger.warning(
                f"[YouTube] InnerTube TV browse FEplaylist_aggregation: "
                f"找不到 playlist renderer，response 中的類型: {sorted(renderer_keys)}"
            )
            raise RuntimeError("InnerTube TV browse 回傳空結果")

        count = 0
        for item in items:
            pl_id = item.get("playlistId")
            if not pl_id:
                continue
            title = self._innertube_extract_text(item.get("title", {}))
            thumbs = item.get("thumbnail", {}).get("thumbnails", [])
            if not thumbs:
                thumbs = item.get("thumbnails", [{}])
                if thumbs and isinstance(thumbs[0], dict) and "thumbnails" in thumbs[0]:
                    thumbs = thumbs[0]["thumbnails"]
            thumb_url = thumbs[-1]["url"] if thumbs else ""
            mapping = ProviderMapping(
                item_id=pl_id, provider_domain=self.domain,
                provider_instance=self.instance_id,
                url=f"https://www.youtube.com/playlist?list={pl_id}",
            )
            pl = Playlist(
                item_id=pl_id, name=title or pl_id,
                provider=self.instance_id,
                provider_mappings={mapping},
            )
            if thumb_url:
                pl.metadata.images = [MediaItemImage(
                    type=ImageType.THUMB, path=thumb_url, provider=self.instance_id,
                )]
            yield pl
            count += 1

        self.logger.info(
            f"[YouTube] InnerTube TV browse 共取得 {count} 個播放清單"
        )

    async def _get_playlists_via_innertube(self) -> AsyncGenerator[Playlist, None]:
        """透過 yt-dlp + cookies 取得 /feed/playlists（包含已儲存他人清單）。

        yt-dlp + cookies 是取得已儲存他人清單的可靠方式。
        """
        if not self._cookies_file:
            raise RuntimeError("取得已儲存他人播放清單需要 cookies.txt")

        # yt-dlp 抓 /feed/playlists（需要 cookies，包含已儲存他人清單）
        url = "https://www.youtube.com/feed/playlists"
        args = (
            ["yt-dlp", "--quiet", "--no-warnings",
             "--flat-playlist", "--dump-json"]
            + self._bgutil_args()
            + ["--cookies", self._cookies_file,
               url]
        )
        self.logger.debug("[YouTube] yt-dlp 取得 /feed/playlists（含已儲存他人清單）…")
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr_data = await proc.communicate()

        if proc.returncode != 0 and not stdout:
            raise RuntimeError(
                f"yt-dlp /feed/playlists 失敗: {stderr_data.decode().strip()[:200]}"
            )

        count = 0
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                playlist = self._json_to_playlist(data)
                if playlist:
                    yield playlist
                    count += 1
            except Exception as exc:
                self.logger.debug(f"[YouTube] 解析 playlist 失敗: {exc}")

        self.logger.info(f"[YouTube] yt-dlp /feed/playlists 共取得 {count} 個播放清單（含已儲存他人清單）")
        if count == 0:
            raise RuntimeError("yt-dlp /feed/playlists 回傳空結果")

    async def _get_playlists_via_ytdlp(self) -> AsyncGenerator[Playlist, None]:
        """透過 yt-dlp + cookies 取得播放清單（fallback）。"""
        url = "https://www.youtube.com/feed/playlists"
        args = (
            ["yt-dlp", "--quiet", "--no-warnings",
             "--flat-playlist", "--dump-json"]
            + self._bgutil_args()
            + self._auth_args()
            + [url]
        )

        self.logger.debug("[YouTube] 透過 yt-dlp 取得播放清單…")
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr_data = await proc.communicate()

            if proc.returncode != 0 and not stdout:
                self.logger.error(
                    f"[YouTube] yt-dlp 取得播放清單失敗: {stderr_data.decode().strip()}"
                )
                return

            count = 0
            for line in stdout.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    playlist = self._json_to_playlist(data)
                    if playlist:
                        yield playlist
                        count += 1
                except Exception as exc:
                    self.logger.debug(f"[YouTube] 解析 playlist 失敗: {exc}")

            self.logger.debug(f"[YouTube] yt-dlp 共取得 {count} 個播放清單")

        except Exception as exc:
            self.logger.error(f"[YouTube] _get_playlists_via_ytdlp 例外: {exc}", exc_info=True)

    def _api_item_to_playlist(self, item: dict) -> Playlist | None:
        """將 YouTube Data API v3 的 playlist item 轉為 MA Playlist 物件。"""
        playlist_id = item.get("id")
        if not playlist_id:
            return None

        snippet = item.get("snippet", {})
        title = snippet.get("title") or playlist_id

        mapping = ProviderMapping(
            item_id=playlist_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=f"https://www.youtube.com/playlist?list={playlist_id}",
        )

        playlist = Playlist(
            item_id=playlist_id,
            name=title,
            provider=self.instance_id,
            provider_mappings={mapping},
            is_editable=False,
        )

        # 封面圖片：取最高解析度
        thumbnails = snippet.get("thumbnails", {})
        thumb_url = None
        for res in ("maxres", "high", "medium", "default"):
            if res in thumbnails:
                thumb_url = thumbnails[res].get("url")
                break
        if thumb_url:
            playlist.metadata.images = [
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=thumb_url,
                    provider=self.instance_id,
                )
            ]

        return playlist

    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """取得單一 playlist 的資訊（不含 tracks）。"""
        await self._ensure_fresh_token()
        return await self._try_methods("取得 playlist",
            [("yt-dlp",    self._get_playlist_via_ytdlp),
             ("InnerTube", self._get_playlist_via_innertube)],
            prov_playlist_id)

    async def _get_playlist_via_api(self, playlist_id: str) -> Playlist:
        """透過 YouTube Data API v3 取得 playlist 資訊。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {"part": "snippet,contentDetails", "id": playlist_id}
        async with self._session.get(
            f"{YOUTUBE_API_BASE}/playlists", headers=headers, params=params,
        ) as resp:
            if resp.status == 401:
                await self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/playlists", headers=headers, params=params,
                ) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()

        items = data.get("items", [])
        if not items:
            raise RuntimeError(f"找不到播放清單: {playlist_id}")

        playlist = self._api_item_to_playlist(items[0])
        if not playlist:
            raise RuntimeError(f"無法解析播放清單: {playlist_id}")
        return playlist

    async def _get_playlist_via_ytdlp(self, playlist_id: str) -> Playlist:
        """透過 yt-dlp 取得 playlist 資訊（fallback）。"""
        url  = f"https://www.youtube.com/playlist?list={playlist_id}"
        args = (
            ["yt-dlp", "--quiet", "--no-warnings",
             "--flat-playlist", "--dump-single-json"]
            + self._bgutil_args()
            + self._auth_args()
            + [url]
        )
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise MediaNotFoundError(
                f"YouTube 播放清單不存在或無法存取: {playlist_id}"
            )
        data     = json.loads(stdout.decode())
        playlist = self._json_to_playlist(data)
        if not playlist:
            raise MediaNotFoundError(f"無法解析播放清單: {playlist_id}")
        return playlist

    async def get_playlist_tracks(
        self, prov_playlist_id: str, page: int = 0
    ) -> list[Track]:
        """
        取得 playlist 內所有 track。

        MA 會以 page=0 呼叫，我們一次回傳全部（yt-dlp 沒有分頁機制）。
        page > 0 時回傳空列表，符合 MA 的分頁慣例。
        """
        if page > 0:
            return []

        await self._ensure_fresh_token()

        # 觀看紀錄假清單
        if prov_playlist_id == "__yt_history__":
            return await self._get_history_tracks()

        is_private = prov_playlist_id.startswith("PLAOYGtd_")

        if is_private:
            # PLAOYGtd_ 是個人私人清單，只能用 yt-dlp + cookies
            if not self._cookies_file:
                raise UnplayableMediaError(
                    f"個人播放清單（{prov_playlist_id}）需要 cookies.txt 才能存取"
                )
            return await self._try_methods(
                f"取得私人播放清單 tracks ({prov_playlist_id})",
                [("yt-dlp+cookies", self._get_playlist_tracks_via_ytdlp)],
                prov_playlist_id
            )

        # 一般公開播放清單：InnerTube → yt-dlp
        return await self._try_methods("取得 playlist tracks",
            [("InnerTube", self._get_playlist_tracks_via_innertube),
             ("yt-dlp",    self._get_playlist_tracks_via_ytdlp)],
            prov_playlist_id)

    async def _get_history_tracks(self) -> list[Track]:
        """透過 yt-dlp + cookies 取得 YouTube 觀看紀錄（最近 50 筆）。"""
        if not self._cookies_file:
            raise UnplayableMediaError("取得觀看紀錄需要設定 cookies.txt")

        url = "https://www.youtube.com/feed/history"
        args = [
            "yt-dlp", "--quiet", "--no-warnings",
            "--flat-playlist", "--dump-json",
            "--playlist-end", "50",
            "--cookies", self._cookies_file,
            url,
        ]
        self.logger.debug("[YouTube] 取得觀看紀錄（最近 50 筆）…")
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr_data = await proc.communicate()

        if proc.returncode != 0 and not stdout:
            raise UnplayableMediaError(
                f"yt-dlp 取得觀看紀錄失敗: {stderr_data.decode().strip()[:200]}"
            )

        tracks: list[Track] = []
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                track = self._to_track(json.loads(line))
                if track:
                    tracks.append(track)
            except Exception as exc:
                self.logger.debug(f"[YouTube] 解析觀看紀錄 track 失敗: {exc}")

        self.logger.info(f"[YouTube] 觀看紀錄共取得 {len(tracks)} 筆")
        return tracks

    async def _get_playlist_tracks_via_ytdlp(self, playlist_id: str) -> list[Track]:
        """透過 yt-dlp 取得播放清單 tracks（fallback）。"""
        url  = f"https://www.youtube.com/playlist?list={playlist_id}"
        # PLAOYGtd_ 是個人私人清單，必須帶 cookies，否則 YouTube 回傳 "playlist does not exist"
        is_private = playlist_id.startswith("PLAOYGtd_")
        if is_private and self._cookies_file:
            auth = ["--cookies", self._cookies_file]
        elif is_private:
            raise RuntimeError("PLAOYGtd_ 私人清單需要 cookies.txt")
        else:
            auth = self._auth_args()
        args = (
            ["yt-dlp", "--quiet", "--no-warnings",
             "--flat-playlist", "--dump-json"]
            + self._bgutil_args()
            + auth
            + [url]
        )
        self.logger.debug(f"[YouTube] yt-dlp 取得 playlist {playlist_id} tracks…")
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr_data = await proc.communicate()

        if proc.returncode != 0 and not stdout:
            self.logger.error(
                f"[YouTube] yt-dlp 取得 playlist tracks 失敗: {stderr_data.decode().strip()}"
            )
            return []

        tracks: list[Track] = []
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                track = self._to_track(json.loads(line))
                if track:
                    tracks.append(track)
            except Exception as exc:
                self.logger.debug(f"[YouTube] 解析 track 失敗: {exc}")

        self.logger.debug(f"[YouTube] yt-dlp playlist {playlist_id} 共 {len(tracks)} 首")
        return tracks

    async def _get_playlist_tracks_via_api(self, playlist_id: str) -> list[Track]:
        """透過 YouTube Data API v3 取得播放清單的 tracks（用於 PLAOYGtd_ 等私人清單）。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params = {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": "50",
        }

        tracks: list[Track] = []
        self.logger.debug(f"[YouTube] 透過 Data API v3 取得 {playlist_id} 的 tracks…")

        try:
            while True:
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/playlistItems",
                    headers=headers,
                    params=params,
                ) as resp:
                    if resp.status == 401:
                        await self._refresh_access_token()
                        headers["Authorization"] = f"Bearer {self._access_token}"
                        continue
                    if resp.status != 200:
                        from aiohttp import ClientResponseError
                        err = await resp.text()
                        raise ClientResponseError(
                            resp.request_info, resp.history,
                            status=resp.status, message=err[:300],
                        )
                    data = await resp.json()

                for item in data.get("items", []):
                    snippet  = item.get("snippet", {})
                    resource = snippet.get("resourceId", {})
                    video_id = resource.get("videoId")
                    if not video_id:
                        continue

                    channel_id   = snippet.get("videoOwnerChannelId", "")
                    channel_name = snippet.get("videoOwnerChannelTitle", "YouTube")
                    raw_title    = snippet.get("title", "Unknown Title")

                    if raw_title in ("Deleted video", "Private video"):
                        continue

                    parsed_title, parsed_artist = self._parse_title(raw_title, channel_name)
                    track_name  = parsed_title or raw_title
                    artist_name = parsed_artist or self._normalize_youtube_topic_channel_name(
                        channel_name
                    )

                    artist_mapping = ProviderMapping(
                        item_id=channel_id or video_id,
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                        url=f"https://www.youtube.com/channel/{channel_id}" if channel_id else "",
                    )
                    artist = Artist(
                        item_id=channel_id or video_id,
                        name=artist_name,
                        provider=self.instance_id,
                        provider_mappings={artist_mapping},
                    )
                    track_mapping = ProviderMapping(
                        item_id=video_id,
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                    )
                    track = Track(
                        item_id=video_id,
                        name=track_name,
                        artists=[artist],
                        provider=self.instance_id,
                        provider_mappings={track_mapping},
                    )
                    thumbnails = snippet.get("thumbnails", {})
                    thumb_url = None
                    for res in ("maxres", "high", "medium", "default"):
                        if res in thumbnails:
                            thumb_url = thumbnails[res].get("url")
                            break
                    if thumb_url:
                        track.metadata.images = [
                            MediaItemImage(
                                type=ImageType.THUMB,
                                path=thumb_url,
                                provider=self.instance_id,
                            )
                        ]
                    tracks.append(track)

                next_page = data.get("nextPageToken")
                if not next_page:
                    break
                params["pageToken"] = next_page

        except Exception as exc:
            self.logger.error(f"[YouTube] _get_playlist_tracks_via_api 例外: {exc}", exc_info=True)

        # 批次補上 duration
        if tracks and self._access_token:
            durations = await self._fetch_video_durations([t.item_id for t in tracks])
            for track in tracks:
                if track.item_id in durations:
                    track.duration = durations[track.item_id]

        self.logger.debug(f"[YouTube] Data API v3 取得 {playlist_id} 共 {len(tracks)} 首")
        return tracks

    async def _fetch_video_durations(self, video_ids: list[str]) -> dict[str, int]:
        """批次透過 Data API v3 取得影片時長，回傳 {video_id: duration_seconds}。"""
        if not video_ids or not self._access_token:
            return {}

        import re
        durations: dict[str, int] = {}
        headers = {"Authorization": f"Bearer {self._access_token}"}

        async def _fetch_batch(batch: list[str]) -> None:
            params = {"part": "contentDetails", "id": ",".join(batch)}
            try:
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/videos", headers=headers, params=params,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                for item in data.get("items", []):
                    vid_id  = item["id"]
                    raw_dur = item.get("contentDetails", {}).get("duration", "")
                    if raw_dur:
                        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", raw_dur)
                        if m:
                            h, mn, s = m.groups()
                            durations[vid_id] = (
                                int(h or 0) * 3600 + int(mn or 0) * 60 + int(s or 0)
                            )
            except Exception as exc:
                self.logger.debug(f"[YouTube] 批次取得 duration 失敗: {exc}")

        # 每批最多 50 個，各批次並行送出
        await asyncio.gather(
            *(_fetch_batch(video_ids[i:i + 50]) for i in range(0, len(video_ids), 50))
        )
        return durations

    def _json_to_playlist(self, data: dict) -> Playlist | None:
        """將 yt-dlp JSON 轉為 MA Playlist 物件。"""
        # yt-dlp flat-playlist 回傳的 playlist 本身 JSON
        # 可能是 {"_type": "playlist", "id": "PLxxx", "title": "..."} 這樣的格式
        # 也可能是 dump-single-json 的完整格式（含 entries）

        playlist_id = data.get("id") or data.get("playlist_id")
        if not playlist_id:
            return None

        # 過濾非 playlist 型別（有些 entry 是 video）
        entry_type = data.get("_type", "")
        if entry_type not in ("playlist", "url", "") and "playlist_id" not in data:
            # 如果是 video entry 而非 playlist，跳過
            if data.get("ie_key", "").lower() not in ("youtubeplaylist", "youtube:tab", ""):
                return None

        title = data.get("title") or data.get("playlist_title") or playlist_id

        mapping = ProviderMapping(
            item_id=playlist_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            url=data.get("webpage_url") or f"https://www.youtube.com/playlist?list={playlist_id}",
        )

        playlist = Playlist(
            item_id=playlist_id,
            name=title,
            provider=self.instance_id,
            provider_mappings={mapping},
            is_editable=False,  # yt-dlp 無法寫入，明確標示唯讀
        )

        # 封面圖片
        thumbnails = data.get("thumbnails")
        thumbnail  = data.get("thumbnail")
        thumb_url  = thumbnails[-1]["url"] if thumbnails else thumbnail
        if thumb_url:
            playlist.metadata.images = [
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=thumb_url,
                    provider=self.instance_id,
                )
            ]

        return playlist

    # ------------------------------------------------------------------
    # 串流
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # InnerTube Player（串流 URL 取得，不需 yt-dlp、不需 cookies）
    # ------------------------------------------------------------------

    # InnerTube 各 client 的設定
    # Android client 的串流 URL 不需要解密 n 參數，且支援年齡限制影片（帶 OAuth）
    _INNERTUBE_CLIENTS = {
        "android_vr": {
            # ANDROID_VR：不需要 PO Token（yt-dlp 預設 client）
            # 注意：clientVersion > 1.65 會觸發 SABR-only 串流實驗，只回傳 f=18
            "context": {
                "client": {
                    "clientName": "ANDROID_VR",
                    "clientVersion": "1.65.10",
                    "deviceMake": "Oculus",
                    "deviceModel": "Quest 3",
                    "androidSdkVersion": 32,
                    "osName": "Android",
                    "osVersion": "12L",
                    "hl": "zh-TW",
                    "gl": "TW",
                    "userAgent": (
                        "com.google.android.apps.youtube.vr.oculus/1.65.10 "
                        "(Linux; U; Android 12L; eureka-user Build/SQ3A.220605.009.A1) gzip"
                    ),
                }
            },
            "api_key": "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
            "headers": {
                "X-YouTube-Client-Name":    "28",
                "X-YouTube-Client-Version": "1.65.10",
                "User-Agent": (
                    "com.google.android.apps.youtube.vr.oculus/1.65.10 "
                    "(Linux; U; Android 12L; eureka-user Build/SQ3A.220605.009.A1) gzip"
                ),
            },
        },
        "android_embedded": {
            # IOS：提供 HLS m3u8，目前不需要 PO Token，支援年齡限制（帶 OAuth）
            "context": {
                "client": {
                    "clientName": "IOS",
                    "clientVersion": "21.02.3",
                    "deviceMake": "Apple",
                    "deviceModel": "iPhone16,2",
                    "osName": "iPhone",
                    "osVersion": "18.3.2.22D82",
                    "hl": "zh-TW",
                    "gl": "TW",
                    "userAgent": (
                        "com.google.ios.youtube/21.02.3 "
                        "(iPhone16,2; U; CPU iOS 18_3_2 like Mac OS X;)"
                    ),
                }
            },
            "api_key": "AIzaSyB-63vPrdThhKuerbB2N_l7Kwwcxj6yUAc",
            "headers": {
                "X-YouTube-Client-Name":    "5",
                "X-YouTube-Client-Version": "21.02.3",
                "User-Agent": (
                    "com.google.ios.youtube/21.02.3 "
                    "(iPhone16,2; U; CPU iOS 18_3_2 like Mac OS X;)"
                ),
            },
        },
        "tv": {
            # TVHTML5：帶 OAuth Bearer 可存取私人/年齡限制影片（等同 SmartTube TV client）
            "context": {
                "client": {
                    "clientName": "TVHTML5",
                    "clientVersion": "7.20260114.12.00",
                    "clientScreen": "WATCH",
                    "hl": "zh-TW",
                    "gl": "TW",
                }
            },
            "api_key": "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
            "headers": {
                "X-YouTube-Client-Name":    "7",
                "X-YouTube-Client-Version": "7.20260114.12.00",
                "User-Agent": (
                    "Mozilla/5.0 (SMART-TV; Linux; Tizen 2.4.0) "
                    "AppleWebKit/538.1 (KHTML, like Gecko) "
                    "Version/2.4.0 SamsungBrowser/1.1 TV Safari/538.1"
                ),
                "Origin": "https://www.youtube.com",
                "Referer": "https://www.youtube.com/tv",
            },
        },
        "android": {
            # ANDROID：備用客戶端
            "context": {
                "client": {
                    "clientName": "ANDROID",
                    "clientVersion": "21.02.35",
                    "androidSdkVersion": 30,
                    "osName": "Android",
                    "osVersion": "11",
                    "hl": "zh-TW",
                    "gl": "TW",
                    "userAgent": (
                        "com.google.android.youtube/21.02.35 "
                        "(Linux; U; Android 11) gzip"
                    ),
                }
            },
            "api_key": "AIzaSyA8eiZmM1FaDVjRy-df2KTyQ_vz_yYM39w",
            "headers": {
                "X-YouTube-Client-Name":    "3",
                "X-YouTube-Client-Version": "21.02.35",
                "User-Agent": (
                    "com.google.android.youtube/21.02.35 "
                    "(Linux; U; Android 11) gzip"
                ),
            },
        },
        "web": {
            # WEB client：支援 Cookie + SAPISIDHASH 認證，用於 cookie 模式取 metadata
            # 注意：WEB client 的串流 URL 有 signatureCipher，不能直接用於串流
            "context": {
                "client": {
                    "clientName": "WEB",
                    "clientVersion": "2.20260114.08.00",
                    "hl": "zh-TW",
                    "gl": "TW",
                }
            },
            "api_key": "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
            "headers": {
                "X-YouTube-Client-Name":    "1",
                "X-YouTube-Client-Version": "2.20260114.08.00",
                "Origin":  "https://www.youtube.com",
                "Referer": "https://www.youtube.com/",
            },
        },
    }

    async def _innertube_player(
        self, video_id: str, client_name: str = "android_vr"
    ) -> dict:
        """呼叫 InnerTube /player 取得影片資訊（含串流 URL）。"""
        import copy
        client_cfg = self._INNERTUBE_CLIENTS[client_name]
        # 新版 InnerTube 不需要 API key，移除 ?key= 避免 400
        url = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false"
        body = {
            "context": copy.deepcopy(client_cfg["context"]),
            "videoId": video_id,
            "contentCheckOk": True,
            "racyCheckOk": True,
        }
        if client_name == "android":
            body["params"] = "CgIQBg=="
        headers = {
            "Content-Type": "application/json",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
            **client_cfg["headers"],
        }
        # 認證策略（每個 client 類型有不同的認證需求）：
        #   TV (TVHTML5) → Bearer token（等同 SmartTube，SAPISIDHASH 無效）
        #   WEB          → Cookie/SAPISIDHASH（Bearer 會 400）
        #   android 系列 → 不注入認證（Cookie/Bearer 都會被拒）
        _ANDROID_CLIENTS = ("android_vr", "android", "android_embedded")
        auth_type = "none"
        if client_name in _ANDROID_CLIENTS:
            auth_type = "none(android)"
        elif client_name == "tv" and self._access_token:
            headers["Authorization"]   = f"Bearer {self._access_token}"
            headers["X-Goog-AuthUser"] = "0"
            auth_type = "Bearer"
        elif client_name == "web" and self._cookies_file:
            headers.update(self._innertube_cookie_headers())
            auth_type = "SAPISIDHASH"
        elif self._access_token:
            headers["Authorization"]   = f"Bearer {self._access_token}"
            headers["X-Goog-AuthUser"] = "0"
            auth_type = "Bearer"
        self.logger.debug(
            f"[YouTube] InnerTube /player {video_id} client={client_name} auth={auth_type}"
        )

        async with self._session.post(url, json=body, headers=headers) as resp:
            if resp.status == 400:
                body_text = await resp.text()
                self.logger.warning(
                    f"[YouTube] InnerTube /player 400 ({client_name}): "
                    f"{body_text[:300]}"
                )
                raise RuntimeError(f"HTTP 400: {body_text[:300]}")
            resp.raise_for_status()
            return await resp.json()

    @staticmethod
    def _pick_best_audio_format(streaming_data: dict) -> dict | None:
        """從 InnerTube streamingData 選出最佳音訊格式。

        優先順序：
        1. adaptiveFormats 中的純音訊（無 video）
        2. 偏好 opus/webm > aac/mp4
        3. 依 bitrate 選最高
        """
        formats: list[dict] = []

        # adaptiveFormats 包含純音訊串流
        for fmt in streaming_data.get("adaptiveFormats", []):
            mime = fmt.get("mimeType", "")
            if "audio" in mime and "video" not in mime:
                formats.append(fmt)

        # 沒有 adaptive 就用 formats（含 video，但音質通常夠）
        if not formats:
            for fmt in streaming_data.get("formats", []):
                mime = fmt.get("mimeType", "")
                if "audio" in mime or "video" in mime:
                    formats.append(fmt)

        if not formats:
            return None

        def score(fmt: dict) -> tuple:
            mime     = fmt.get("mimeType", "")
            bitrate  = fmt.get("bitrate", 0) or fmt.get("averageBitrate", 0) or 0
            is_opus  = "opus" in mime
            is_aac   = "aac" in mime or "mp4a" in mime
            return (is_opus, is_aac, bitrate)

        return max(formats, key=score)

    @staticmethod
    def _innertube_mime_to_content_type(mime: str) -> str:
        """把 InnerTube mimeType 轉成 MA ContentType。"""
        mime = mime.lower()
        if "opus" in mime or "webm" in mime:
            return "ogg"   # opus in webm container
        if "mp4a" in mime or "aac" in mime or "mp4" in mime:
            return "m4a"
        return "unknown"

    _innertube_fail_until: float = 0.0  # InnerTube 連續失敗時暫時跳過的截止時間
    _INNERTUBE_COOLDOWN: float = 600  # InnerTube 失敗冷卻 10 分鐘

    async def _get_stream_url_via_innertube(self, video_id: str) -> tuple[str, str, int, str, int]:
        """透過 InnerTube 取得串流直連 URL。

        回傳 (url, content_type, bitrate, client_name, duration_seconds)。

        策略：並行嘗試多個 client，取第一個成功的結果。
        android_vr / android 不需 PO Token，成功率最高。
        ios 可取得 URL 但被 bot 標記時下載 403，作為最後手段並驗證 URL。
        web 的串流 URL 有 signatureCipher，不用於串流。
        tv 匿名時幾乎總是 UNPLAYABLE/LOGIN_REQUIRED。
        """
        if self._access_token or self._cookies_file:
            clients_to_try = ["android_vr", "android", "tv"]
        else:
            clients_to_try = ["android_vr", "android"]

        async def _try_client(client_name: str) -> tuple[str, dict] | None:
            """嘗試單一 client，成功回傳 (client_name, fmt_dict, data)，失敗回傳 None。"""
            try:
                data = await self._innertube_player(video_id, client_name)
            except Exception as exc:
                return ("error", client_name, str(exc)[:60], None)

            status = data.get("playabilityStatus", {}).get("status", "")
            reason = data.get("playabilityStatus", {}).get("reason", "")

            if status != "OK":
                self.logger.debug(
                    f"[YouTube] {video_id} client={client_name} "
                    f"status={status} reason={reason!r}"
                )
                return ("fail", client_name, f"status={status} reason={reason}", data)

            streaming_data = data.get("streamingData", {})
            fmt = self._pick_best_audio_format(streaming_data)
            if not fmt or not fmt.get("url"):
                return ("fail", client_name, "無直連音訊 URL", data)

            return ("ok", client_name, fmt, data)

        tasks = [_try_client(c) for c in clients_to_try]
        results = await asyncio.gather(*tasks)

        errors: list[str] = []
        for r in results:
            if r is None:
                continue
            tag, client_name = r[0], r[1]
            if tag == "ok":
                fmt, data = r[2], r[3]
                stream_url = fmt["url"]
                mime       = fmt.get("mimeType", "")
                bitrate    = fmt.get("bitrate", 0) or fmt.get("averageBitrate", 0) or 128000
                ct         = self._innertube_mime_to_content_type(mime)
                inn_dur    = int(data.get("videoDetails", {}).get("lengthSeconds", 0) or 0)
                self.logger.info(
                    f"[YouTube] InnerTube player 成功 ({client_name}): "
                    f"{video_id} mime={mime} bitrate={bitrate//1000}kbps"
                )
                return stream_url, ct, bitrate, client_name, inn_dur
            else:
                errors.append(f"{client_name}: {r[2]}")

        err_detail = " | ".join(errors)
        raise UnplayableMediaError(
            f"InnerTube 無法取得串流 URL: {video_id} - {err_detail}"
        )

    _FMT_PREFERENCE = "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best"

    def _ensure_ytdlp(self) -> Any:
        """懶載入 yt_dlp 模組（從系統 site-packages），避免每次 subprocess 啟動開銷。"""
        if self._yt_dlp_module is None:
            import sys
            sp = "/usr/local/lib/python3.13/site-packages"
            if sp not in sys.path:
                sys.path.insert(0, sp)
            import yt_dlp
            self._yt_dlp_module = yt_dlp
            self.logger.debug("[YouTube] yt_dlp 模組已載入（Python API 模式）")
        return self._yt_dlp_module

    async def _get_direct_url(self, item_id: str, url: str) -> dict:
        """透過 yt-dlp Python API 取得直連 URL 和格式資訊。

        回傳 dict: {url, content_type, bitrate, sample_rate, duration}。
        使用 in-process Python API（省去 ~1s subprocess 啟動開銷），
        含記憶體快取（TTL 5h）。
        """
        cached = self._direct_url_cache.get(item_id)
        if cached:
            expire_ts, result = cached
            if time.time() < expire_ts:
                self.logger.debug(f"[YouTube] yt-dlp URL 快取命中: {item_id}")
                return result
            del self._direct_url_cache[item_id]

        yt_dlp = self._ensure_ytdlp()

        def _build_opts(fast_client: str = "") -> dict:
            opts: dict = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "format": self._FMT_PREFERENCE,
                "nocheckcertificate": True,
            }
            ea: dict = {}
            if fast_client:
                ea.setdefault("youtube", {})["player_client"] = [fast_client]
            if self._cookies_file:
                opts["cookiefile"] = self._cookies_file
            if self._access_token:
                opts.setdefault("http_headers", {})[
                    "Authorization"
                ] = f"{self._token_type} {self._access_token}"
            if self._bgutil_url:
                default_urls = {"http://127.0.0.1:4416", "http://localhost:4416"}
                if self._bgutil_url not in default_urls:
                    ea.setdefault("youtubepot-bgutilhttp", {})["base_url"] = [self._bgutil_url]
            if ea:
                opts["extractor_args"] = ea
            return opts

        def _extract(opts: dict) -> dict | None:
            """同步呼叫 yt-dlp extract_info，回傳解析後的 dict 或 None。"""
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            except Exception:
                return None
            if not info or not info.get("url", "").startswith("http"):
                return None

            ext = info.get("ext", "")
            ct = ContentType.UNKNOWN
            if ext in ("webm", "opus"):
                ct = ContentType.OGG
            elif ext in ("m4a", "aac", "mp4"):
                ct = ContentType.M4A
            elif ext == "mp3":
                ct = ContentType.MP3

            abr = info.get("abr") or 0
            bitrate = int(float(abr) * 1000) if abr else 0
            sample_rate = int(info.get("asr") or 44100)
            duration = int(info.get("duration") or 0)

            return {
                "url": info["url"],
                "content_type": ct,
                "bitrate": bitrate,
                "sample_rate": sample_rate,
                "duration": duration,
            }

        def _cache_and_return(res: dict) -> dict:
            # 快取過大時先清掉過期項目，避免長時間運行後記憶體無限增長
            if len(self._direct_url_cache) > 500:
                now_ts = time.time()
                for key in [
                    k for k, v in self._direct_url_cache.items() if v[0] <= now_ts
                ]:
                    del self._direct_url_cache[key]
            self._direct_url_cache[item_id] = (
                time.time() + self._DIRECT_URL_CACHE_TTL, res
            )
            return res

        loop = asyncio.get_running_loop()

        # --- 快速路徑：android_vr（~1.5s），冷卻期內自動跳過 ---
        now = time.time()
        if now >= self._fast_client_fail_until:
            opts_fast = _build_opts(fast_client="android_vr")
            result = await loop.run_in_executor(None, _extract, opts_fast)
            if result:
                return _cache_and_return(result)
            self._fast_client_fail_until = now + self._FAST_CLIENT_COOLDOWN
            self.logger.debug(
                f"[YouTube] yt-dlp android_vr 失敗，冷卻 {self._FAST_CLIENT_COOLDOWN:.0f}s，"
                f"回退 bgutil 流程: {item_id}"
            )
        else:
            self.logger.debug(
                f"[YouTube] android_vr 冷卻中，直接走 bgutil 流程: {item_id}"
            )

        # --- 回退路徑：預設流程（~4s，bgutil PO Token + 多 client fallback）---
        opts_default = _build_opts()
        result = await loop.run_in_executor(None, _extract, opts_default)
        if result:
            self._fast_client_fail_until = 0.0
            self._innertube_fail_until = 0.0
            return _cache_and_return(result)

        self.logger.warning(f"[YouTube] 無法取得直連 URL ({item_id})")
        raise UnplayableMediaError(f"YouTube 影片無法播放: {item_id}")

    def _patch_queue_item_duration(self, item_id: str, dur: int) -> None:
        """回補 QueueItem.duration 和 media_item.duration。

        MASS 的 seek() 檢查 queue.current_item.duration（非 streamdetails.duration），
        LRCLIB 檢查 track.duration（即 media_item.duration）。
        _load_item() 在 get_stream_details() 之後才設定 queue.current_item，
        因此必須遍歷 _queue_items 全清單才能找到正在載入的 item。
        """
        if not dur:
            return
        try:
            pq = self.mass.player_queues
            for queue in pq:
                for qi in pq.items(queue.queue_id, 500, 0):
                    if qi.duration or not qi.media_item:
                        continue
                    for pm in qi.media_item.provider_mappings:
                        if pm.item_id == item_id:
                            qi.duration = dur
                            qi.media_item.duration = dur
                            break
        except Exception:
            pass

    def _start_prefetch(self, current_item_id: str) -> None:
        """啟動背景預取（取消之前尚未完成的預取 task）。"""
        import asyncio

        if self._prefetch_task and not self._prefetch_task.done():
            self._prefetch_task.cancel()
        self._prefetch_task = asyncio.create_task(
            self._prefetch_next_queue_items(current_item_id)
        )

    async def _prefetch_next_queue_items(self, current_item_id: str) -> None:
        """背景預取佇列中接下來幾首 YouTube 歌曲的直連 URL 到快取。

        在 get_stream_details 完成後透過 asyncio.create_task 啟動，
        讓 MA 的標準預載器後續呼叫 get_stream_details 時直接命中快取（~0ms）。
        """
        try:
            pq = self.mass.player_queues
            upcoming_ids: list[str] = []

            for queue in pq:
                all_items = pq.items(queue.queue_id, 500, 0)
                current_idx = -1
                for i, qi in enumerate(all_items):
                    if not qi.media_item:
                        continue
                    for pm in qi.media_item.provider_mappings:
                        if pm.item_id == current_item_id and (
                            pm.provider_domain == self.domain
                            or pm.provider_instance == self.instance_id
                        ):
                            current_idx = i
                            break
                    if current_idx >= 0:
                        break

                if current_idx < 0:
                    continue

                for qi in all_items[current_idx + 1:]:
                    if len(upcoming_ids) >= self._PREFETCH_AHEAD:
                        break
                    if not qi.media_item:
                        continue
                    for pm in qi.media_item.provider_mappings:
                        if pm.provider_domain == self.domain or pm.provider_instance == self.instance_id:
                            vid = pm.item_id
                            cached = self._direct_url_cache.get(vid)
                            if cached and time.time() < cached[0]:
                                continue
                            if vid not in self._prefetch_in_progress:
                                upcoming_ids.append(vid)
                            break
                if upcoming_ids:
                    break

            if not upcoming_ids:
                return

            self.logger.info(
                "[YouTube] 背景預取 %d 首: %s",
                len(upcoming_ids),
                ", ".join(upcoming_ids),
            )

            for vid in upcoming_ids:
                self._prefetch_in_progress.add(vid)
                try:
                    yt_url = f"https://www.youtube.com/watch?v={vid}"
                    await self._get_direct_url(vid, yt_url)
                    self.logger.info("[YouTube] 預取完成: %s", vid)
                except Exception as exc:
                    self.logger.debug("[YouTube] 預取失敗 %s: %s", vid, exc)
                finally:
                    self._prefetch_in_progress.discard(vid)

        except Exception as exc:
            self.logger.debug("[YouTube] 佇列預取錯誤: %s", exc)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """取得串流詳情。

        策略：InnerTube 與 yt-dlp 並行競速，誰先成功回傳就用誰；
        若一方失敗則等待另一方；兩方皆失敗才拋出例外。
        InnerTube 冷卻中時直接只走 yt-dlp。
        """
        await self._ensure_fresh_token()

        yt_url = f"https://www.youtube.com/watch?v={item_id}"
        now = time.time()

        async def _innertube_path() -> tuple[StreamDetails, int]:
            stream_url, ct_str, bitrate, client_name, inn_dur = (
                await self._get_stream_url_via_innertube(item_id)
            )
            try:
                content_type = ContentType(ct_str)
            except ValueError:
                content_type = ContentType.UNKNOWN
            self.logger.info(
                f"[YouTube] InnerTube 串流 ({client_name}): {item_id} "
                f"bitrate={bitrate // 1000}kbps"
            )
            self._innertube_fail_until = 0.0
            return StreamDetails(
                item_id=item_id,
                provider=self.instance_id,
                audio_format=AudioFormat(
                    content_type=content_type,
                    sample_rate=44100,
                    bit_depth=16,
                    channels=2,
                    bit_rate=bitrate,
                ),
                stream_type=StreamType.HTTP,
                path=stream_url,
                duration=inn_dur or None,
                can_seek=True,
                allow_seek=True,
                data={"source": "innertube", "client": client_name},
            ), inn_dur

        async def _ytdlp_path() -> tuple[StreamDetails, int]:
            info = await self._get_direct_url(item_id, yt_url)
            self.logger.debug(
                f"[YouTube] yt-dlp 取得串流: {item_id} "
                f"ct={info['content_type']} br={info['bitrate']//1000}kbps"
            )
            return StreamDetails(
                item_id=item_id,
                provider=self.instance_id,
                audio_format=AudioFormat(
                    content_type=info["content_type"],
                    sample_rate=info["sample_rate"],
                    bit_depth=16,
                    channels=2,
                    bit_rate=info["bitrate"],
                ),
                stream_type=StreamType.HTTP,
                path=info["url"],
                duration=info["duration"] or None,
                can_seek=True,
                allow_seek=True,
                data={"source": "ytdlp"},
            ), info["duration"]

        # InnerTube 冷卻中：只走 yt-dlp
        if now < self._innertube_fail_until:
            self.logger.debug(f"[YouTube] InnerTube 冷卻中，直接走 yt-dlp: {item_id}")
            try:
                sd, dur = await _ytdlp_path()
                self._patch_queue_item_duration(item_id, dur)
                self._start_prefetch(item_id)
                return sd
            except Exception as e:
                raise UnplayableMediaError(f"YouTube 影片無法播放: {item_id} - {e}") from e

        # yt-dlp 路徑錯開啟動：run_in_executor 一旦開跑便無法中途取消，
        # InnerTube 通常 <1s 成功，稍加延遲可避免每次播放都白跑一次完整 yt-dlp 解析
        # （省下 executor 執行緒與對 YouTube 的多餘請求）。
        # 快取已命中（背景預取）時不延遲，維持 ~0ms 直接回傳。
        cached_url = self._direct_url_cache.get(item_id)
        has_url_cache = bool(cached_url and now < cached_url[0])

        async def _ytdlp_path_staggered() -> tuple[StreamDetails, int]:
            if not has_url_cache:
                await asyncio.sleep(1.5)
            return await _ytdlp_path()

        # 並行競速：誰先成功就用誰
        tasks: dict[asyncio.Task, str] = {
            asyncio.ensure_future(_innertube_path()): "innertube",
            asyncio.ensure_future(_ytdlp_path_staggered()): "ytdlp",
        }
        errors: dict[str, BaseException] = {}
        pending = set(tasks.keys())

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is None:
                    # 取消尚未完成的另一條路徑
                    for p in pending:
                        p.cancel()
                        try:
                            await p
                        except (asyncio.CancelledError, Exception):
                            pass
                    sd, dur = task.result()
                    self._patch_queue_item_duration(item_id, dur)
                    self._start_prefetch(item_id)
                    return sd
                else:
                    name = tasks[task]
                    errors[name] = exc
                    if name == "innertube":
                        t = time.time()
                        self._innertube_fail_until = t + self._INNERTUBE_COOLDOWN
                        self._fast_client_fail_until = t + self._INNERTUBE_COOLDOWN
                        self.logger.debug(f"[YouTube] InnerTube 失敗，等待 yt-dlp: {exc}")

        raise UnplayableMediaError(
            f"YouTube 影片無法播放: {item_id} - "
            f"innertube={errors.get('innertube')} | ytdlp={errors.get('ytdlp')}"
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    async def _video_info_via_innertube(self, video_id: str) -> dict:
        """透過 InnerTube /player 取得影片基本資訊（metadata）。"""
        # 客戶端優先順序：
        #   有 cookies → WEB 最可靠（cookie auth，支援所有影片類型）
        #   僅 OAuth   → TV 帶 Bearer（但音樂影片經常 UNPLAYABLE）
        #   匿名       → android_vr（不需 PO Token）
        if self._cookies_file:
            clients_to_try = ("web",)
        elif self._access_token:
            clients_to_try = ("tv",)
        else:
            clients_to_try = ("android_vr", "android")
        for client in clients_to_try:
            try:
                data = await self._innertube_player(video_id, client)
            except Exception as exc:
                self.logger.debug(f"[YouTube] _video_info_via_innertube ({client}) exception: {exc}")
                continue
            status = data.get("playabilityStatus", {}).get("status", "")
            details = data.get("videoDetails", {})
            title   = details.get("title", "")
            self.logger.debug(
                f"[YouTube] _video_info_via_innertube ({client}) {video_id}: "
                f"status={status!r} title={title!r}"
            )
            if title:
                dur    = int(details.get("lengthSeconds", "0") or "0")
                thumbs = details.get("thumbnail", {}).get("thumbnails", [])
                self.logger.debug(
                    f"[YouTube] _video_info_via_innertube ({client}): {video_id} title={title!r}"
                )
                return {
                    "id":          video_id,
                    "title":       title,
                    "channel":     details.get("author", ""),
                    "channel_id":  details.get("channelId", ""),
                    "duration":    dur,
                    "thumbnail":   thumbs[-1]["url"] if thumbs else "",
                    "description": details.get("shortDescription", ""),
                }
        raise RuntimeError(f"InnerTube /player 無法取得影片 metadata: {video_id}")

    async def _video_info(self, video_id: str) -> dict:
        """取得影片資訊。Data API → InnerTube → yt-dlp。"""
        await self._ensure_fresh_token()
        methods: list[tuple] = []
        if self._is_api_available():
            methods.append(("DataAPI", self._video_info_via_api))
        methods.append(("InnerTube", self._video_info_via_innertube))
        methods.append(("yt-dlp", self._video_info_via_ytdlp))
        return await self._try_methods("取得影片資訊", methods, video_id)

    async def _video_info_via_api(self, video_id: str) -> dict:
        """透過 YouTube Data API v3 取得影片資訊。"""
        headers = {"Authorization": f"Bearer {self._access_token}"}
        params  = {
            "part": "snippet,contentDetails,statistics",
            "id":   video_id,
        }
        async with self._session.get(
            f"{YOUTUBE_API_BASE}/videos",
            headers=headers,
            params=params,
        ) as resp:
            if resp.status == 401:
                await self._refresh_access_token()
                headers["Authorization"] = f"Bearer {self._access_token}"
                async with self._session.get(
                    f"{YOUTUBE_API_BASE}/videos",
                    headers=headers,
                    params=params,
                ) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()

        items = data.get("items", [])
        if not items:
            raise RuntimeError(f"YouTube Data API v3 找不到影片: {video_id}")

        item    = items[0]
        snippet = item.get("snippet", {})
        details = item.get("contentDetails", {})

        # 解析 duration（ISO 8601 格式，例如 PT3M45S）
        duration = 0
        raw_dur = details.get("duration", "")
        if raw_dur:
            import re
            m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", raw_dur)
            if m:
                h, mn, s = m.groups()
                duration = int(h or 0) * 3600 + int(mn or 0) * 60 + int(s or 0)

        # 縮圖
        thumbnails_raw = snippet.get("thumbnails", {})
        thumbnails = []
        for res in ("maxres", "high", "medium", "default"):
            if res in thumbnails_raw:
                thumbnails.append({"url": thumbnails_raw[res]["url"]})

        return {
            "id":          video_id,
            "title":       snippet.get("title", ""),
            "uploader":    snippet.get("channelTitle", ""),
            "channel":     snippet.get("channelTitle", ""),
            "channel_id":  snippet.get("channelId", ""),
            "uploader_id": snippet.get("channelId", ""),
            "duration":    duration,
            "thumbnail":   thumbnails[0]["url"] if thumbnails else None,
            "thumbnails":  thumbnails,
            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
        }

    async def _video_info_via_ytdlp(self, video_id: str) -> dict:
        """透過 yt-dlp 取得影片資訊（fallback）。"""
        args = (
            ["yt-dlp", "--dump-json", "--no-playlist"]
            + self._bgutil_args()
            + self._ejs_args()
            + self._auth_args()
            + [f"https://www.youtube.com/watch?v={video_id}"]
        )
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp 錯誤: {stderr.decode().strip()}")
        return json.loads(stdout.decode())
