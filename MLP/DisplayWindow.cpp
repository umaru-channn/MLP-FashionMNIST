// DisplayWindow.cpp
// グリッド画像の描画、詳細ビュー（拡大画像 + Top-10表示）
// 進捗バーなど GUI 表示を一括管理するモジュール
// Win32 API によりウィンドウを生成し、描画を担当する
// CIFAR-10対応: RGB カラー画像をサポート
#include "DisplayWindow.h"
#include <windows.h>
#include <vector>
#include <string> 
#include <utility>
#include <algorithm>

// グローバル変数
// ウィンドウハンドル / フレームバッファ / ラベル情報など
// ウィンドウハンドル（描画ターゲット）
static HWND g_hWnd = nullptr;
// フレームバッファ（左側のグリッド画像用）RGBA
static std::vector<uint8_t> g_framebuffer;
// フレームバッファの幅（ピクセル）
static int g_framebufferWidth = 0;
// フレームバッファの高さ（ピクセル）
static int g_fbHeight = 0;
// 進捗バーの進捗値（0.0～1.0）
static float g_progress = 0.0f;
// 正解ラベル配列（100件）
static std::vector<int> g_gtLabels;
// 予測ラベル配列（100件）
static std::vector<int> g_predLabels;
// 正誤判定（true=正解 / false=不正解）
static std::vector<bool> g_correctFlags;
// グリッド画像 1枚の横幅（32 for CIFAR-10）
static int g_imageWidth = 32;
// グリッド画像 1枚の縦高さ（32 for CIFAR-10）
static int g_imageHeight = 32;
// グリッド描画列数（10列など）
static int g_gridColumns = 10;
// グリッド画像の拡大スケール
static int g_scale = 1;
// グリッド画像下に描画するラベル領域の高さ
static int g_labelH = 18;
// グリッド画像横のマージン（画像間スペース）
static int g_marginX = 34;
// グリッド画像縦のマージン（画像間スペース）
static int g_marginY = 12;
// RGB画像かどうか（true: RGB 3チャンネル, false: グレースケール 1チャンネル）
static bool g_isRGB = true;
// 右側詳細ビュー 用データ保持
// 拡大表示する1枚画像（32×32×3 for CIFAR-10）
static std::vector<uint8_t> g_detailImage;
// 拡大画像の幅（32 for CIFAR-10）
static int g_detailImgW = 32;
// 拡大画像の高さ（32 for CIFAR-10）
static int g_detailImgH = 32;
// Top-10 の (classID, probability) の配列
static std::vector<std::pair<int, float>> g_top10;
// Top-10 のクラス名（"airplane" など）
static std::vector<std::wstring> g_top10Names;

// 中央配置のためのオフセット計算（グリッド全体を中央に置く）
void CalcCenteredOffset(int winW, int winH, int& outX, int& outY)
{
// フレームバッファを中央に置くようにオフセットを計算する
//outX = (winW - g_framebufferWidth) / 2;
// 任意の位置(20)に設定する
outX = 20;   // ← 左から20pxに配置
// 下側には進捗バーがあるので 30px 余裕
outY = (winH - g_fbHeight - 30) / 2;

// もし画面外になったら補正する
if (outX < 0) outX = 0;
if (outY < 0) outY = 0;
}

// フレームバッファ（左側グリッド画像領域）を構築する
// 100枚の画像を RGBA の g_framebuffer に敷き詰める
// CIFAR-10対応: RGB画像をサポート
void BuildFramebuffer(const std::vector<std::vector<uint8_t>>& images)
{
// グリッド画像の枚数（通常100）
int count = (int)images.size();
// 1画像のスケール後の横幅（例：32 × 2）
int scaledWidth = g_imageWidth * g_scale;
// 1画像のスケール後の縦幅
int scaledHeight = g_imageHeight * g_scale;
// 列数（10列など）
int columns = g_gridColumns;
// 行数（100 / 10 = 10行）
int rows = (count + columns - 1) / columns;
// フレームバッファ全体の横幅（画像幅 + マージン）
g_framebufferWidth = columns * (scaledWidth + g_marginX);
// フレームバッファ全体の縦幅（画像 + ラベル高さ + 縦マージン）
g_fbHeight = rows * (scaledHeight + g_labelH + g_marginY);
// RGBA フレームバッファ確保（幅×高さ×4）
g_framebuffer.assign(g_framebufferWidth * g_fbHeight * 4, 0);

// 画像がRGBかどうかを自動判定（32×32×3 = 3072バイト or 32×32 = 1024バイト）
if (!images.empty()) {
size_t expectedRGBSize = static_cast<size_t>(g_imageWidth) * g_imageHeight * 3;
g_isRGB = (images[0].size() == expectedRGBSize);
}

// すべての画像をグリッドに敷き詰めるループ
for (int i = 0; i < count; i++)
{
// 画像の X グリッド位置
int gridX = i % columns;
// 画像の Y グリッド位置
int gridY = i / columns;
// この画像の描画開始位置 X（左上位置）
int baseX = gridX * (scaledWidth + g_marginX);
// この画像の描画開始位置 Y（上部 + ラベル分の余白）
int baseY = gridY * (scaledHeight + g_labelH + g_marginY);
// 対象画像
const auto& imageData = images[i];
// 画像の全ピクセルを描く
for (int y = 0; y < g_imageHeight; ++y)
{
for (int x = 0; x < g_imageWidth; ++x)
{
uint8_t r, g, b;
if (g_isRGB) {
// RGB画像の場合: インターリーブ形式 (R,G,B,R,G,B...)
int pixelIndex = y * g_imageWidth + x;
r = imageData[pixelIndex * 3 + 0];
g = imageData[pixelIndex * 3 + 1];
b = imageData[pixelIndex * 3 + 2];
} else {
// グレースケール画像の場合
uint8_t v = imageData[y * g_imageWidth + x];
r = g = b = v;
}
// スケール倍率で拡大描画する
for (int yy = 0; yy < g_scale; yy++)
{
for (int xx = 0; xx < g_scale; xx++)
{
// 拡大先の X 座標
int dx = baseX + x * g_scale + xx;
// 拡大先の Y 座標
int dy = baseY + y * g_scale + yy;
// フレームバッファ範囲外ならスキップ
if (dx < 0 || dy < 0 || dx >= g_framebufferWidth || dy >= g_fbHeight)
continue;
// フレームバッファのインデックス
int idx = (dy * g_framebufferWidth + dx) * 4;
// RGB → BGR順 (Windows DIB形式) で書き込み（A=255）
g_framebuffer[idx + 0] = b;  // Blue
g_framebuffer[idx + 1] = g;  // Green
g_framebuffer[idx + 2] = r;  // Red
g_framebuffer[idx + 3] = 255; // Alpha
}
}
}
}
}
}

// グリッド画像下のラベル（GT/Pred/O/X）描画
void DrawLabels(HDC hdc, int ox, int oy)
{
// ラベル用のフォント作成（太字 14pt）
HFONT font = CreateFontW(
14, 0, 0, 0,          // 高さ=14
FW_BOLD,              // 太字
FALSE, FALSE, FALSE,
DEFAULT_CHARSET,
OUT_DEFAULT_PRECIS,
CLIP_DEFAULT_PRECIS,
DEFAULT_QUALITY,
FIXED_PITCH,
L"Consolas"
);
// 古いフォントを退避
HFONT oldFont = (HFONT)SelectObject(hdc, font);
// 背景透過
SetBkMode(hdc, TRANSPARENT);
// g_gtLabels の数（通常100）
int count = (int)g_gtLabels.size();
// スケール後の幅/高さ
int scaledWidth = g_imageWidth * g_scale;
int scaledHeight = g_imageHeight * g_scale;
// 全画像分ループ
for (int i = 0; i < count; i++)
{
// グリッド座標（何列・何行）
int gx = i % g_gridColumns;
int gy = i / g_gridColumns;
// 描画開始位置（画面座標）
int baseX = ox + gx * (scaledWidth + g_marginX);
int baseY = oy + gy * (scaledHeight + g_labelH + g_marginY);
// ラベル値取得
int gt = g_gtLabels[i];
int pr = g_predLabels[i];
bool ok = g_correctFlags[i];
// 表示する文字列を作る
wchar_t buf[64];
swprintf(buf, 64, L"GT:%d Pred:%d", gt, pr);
// 正解なら緑 / 不正解なら赤
SetTextColor(hdc, ok ? RGB(0, 255, 0) : RGB(255, 80, 80));
// 上記のラベルを描画
TextOutW(hdc, baseX + 4, baseY + scaledHeight + 2, buf, (int)wcslen(buf));
}

// 古いフォントに戻す
SelectObject(hdc, oldFont);
DeleteObject(font);
}

// 進捗バーを描画する（ウィンドウ下部）
void DrawProgressBar(HDC hdc, int winW, int winH)
{
// 背景（濃い灰色）
RECT bg = { 0, winH - 30, winW, winH };
::FillRect(hdc, &bg, (HBRUSH)GetStockObject(DKGRAY_BRUSH));
// 進捗の太さ
int barW = (int)(winW * g_progress);
// 緑色バー
RECT bar = { 0, winH - 30, barW, winH };
HBRUSH green = CreateSolidBrush(RGB(0, 255, 0));
::FillRect(hdc, &bar, green);
DeleteObject(green);
}

// 詳細ビュー（拡大画像 + Top-10）描画処理
// ・ウィンドウ右側に配置（リサイズ対応）
// ・拡大画像は 6倍（192×192 for 32×32）
// ・正解 → 緑枠、不正解 → 赤枠（×なし）
// ・Top-10 の文字表示 ＋ 横棒グラフ（追加）
// CIFAR-10対応: RGBカラー画像をサポート
void DrawDetailView(HDC hdc, int ox, int oy, int winW, int winH)
{
// 必要データが揃っていなければ描画しない
if (g_detailImage.empty() || g_top10.empty())
return;
// 拡大倍率（6倍）
const int scale = 6;
// 元画像（32×32）の幅・高さ
int imgW = g_detailImgW;
int imgH = g_detailImgH;
// 拡大表示サイズ（192×192 for 32×32）
int drawW = imgW * scale;
int drawH = imgH * scale;
// 描画X開始（左グリッドの右横 + 20px）
int startX = ox + g_framebufferWidth + 20;
// ウィンドウ右端から20pxはみ出さないよう調整
int desiredRightMargin = 20;
int maxX = winW - desiredRightMargin - drawW;
if (startX > maxX) startX = maxX;
if (startX < 0)    startX = 0;
// 描画Y開始（グリッドと同じ）
int startY = oy;
// 進捗バーを避けるため、下から60px確保
int desiredBottomMargin = 60;
int maxY = winH - desiredBottomMargin - drawH;
if (startY > maxY) startY = maxY;
if (startY < 0)    startY = 0;

// 詳細画像がRGBかどうかを判定
size_t expectedRGBSize = static_cast<size_t>(imgW) * imgH * 3;
bool detailIsRGB = (g_detailImage.size() == expectedRGBSize);

// 拡大画像を1ピクセルずつ描画
for (int y = 0; y < imgH; ++y)
{
for (int x = 0; x < imgW; ++x)
{
uint8_t r, g, b;
if (detailIsRGB) {
// RGB画像の場合
int pixelIndex = y * imgW + x;
r = g_detailImage[pixelIndex * 3 + 0];
g = g_detailImage[pixelIndex * 3 + 1];
b = g_detailImage[pixelIndex * 3 + 2];
} else {
// グレースケール画像の場合
uint8_t v = g_detailImage[y * imgW + x];
r = g = b = v;
}
// カラーブラシ作成
HBRUSH br = CreateSolidBrush(RGB(r, g, b));
// 拡大後の1ピクセル分の矩形
RECT rc{ startX + x * scale, startY + y * scale, startX + (x + 1) * scale, startY + (y + 1) * scale };
// ピクセル描画
::FillRect(hdc, &rc, br);
// ブラシ破棄
DeleteObject(br);
}
}
// 対象画像が正解かどうか判定（0番目が対象データ）
bool isCorrect = (!g_correctFlags.empty()) ? g_correctFlags[0] : false;
// 枠色：正解→緑、不正解→赤
COLORREF frameColor = isCorrect ? RGB(60, 255, 60) : RGB(255, 60, 60);
// 4px のペン生成
HPEN pen = CreatePen(PS_SOLID, 4, frameColor);
// 元の描画設定を保存
HPEN oldPen = (HPEN)SelectObject(hdc, pen);
HBRUSH oldBrush = (HBRUSH)SelectObject(hdc, GetStockObject(NULL_BRUSH));
// 拡大画像の周囲に枠を描く（×は描かない）
Rectangle(hdc, startX - 2, startY - 2, startX + drawW + 2, startY + drawH + 2);
// 元に戻す
SelectObject(hdc, oldPen);
SelectObject(hdc, oldBrush);
DeleteObject(pen);
// Top-10 描画開始位置（画像の下）
int textStartX = startX;
int textStartY = startY + drawH + 10;
// Top-10 表示用フォント作成
HFONT font = CreateFontW(
15, 0, 0, 0,
FW_BOLD, FALSE, FALSE, FALSE,
DEFAULT_CHARSET,
OUT_DEFAULT_PRECIS,
CLIP_DEFAULT_PRECIS,
DEFAULT_QUALITY,
FIXED_PITCH,
L"Consolas"
);
// フォント設定
HFONT oldFont = (HFONT)SelectObject(hdc, font);
SetBkMode(hdc, TRANSPARENT);
SetTextColor(hdc, RGB(255, 255, 255));
// 横棒グラフの基本設定
const int barWidth = 240; // 横幅
const int barHeight = 14;  // 縦の太さ
const int barGap = 20;  // 行間
// グラフ背景（灰色）
HBRUSH emptyBrush = CreateSolidBrush(RGB(70, 70, 70));
// グラフの塗り部分（青）
HBRUSH filledBrush = CreateSolidBrush(RGB(60, 150, 255));
// Top-10 をループで描画（文字 + 横棒グラフ）
for (int i = 0; i < (int)g_top10.size(); ++i)
{
// クラス ID と 確率
int cls = g_top10[i].first;
float prob = g_top10[i].second;
// クラス名（存在しなければ "?"）
const wchar_t* cname = (i < (int)g_top10Names.size()) ? g_top10Names[i].c_str() : L"?";
// 描画用文字列の組み立て
wchar_t buf[128];
swprintf(buf, 128, L"%d: %s (%.1f%%)", cls, cname, prob * 100.0f);
// この行の縦位置
int yloc = textStartY + i * (barHeight + barGap);
// 既存の文字表示
TextOutW(hdc, textStartX, yloc, buf, (int)wcslen(buf));
// 横棒グラフ表示
// 背景バー（灰色）
RECT bgRect{ textStartX, yloc + 18, textStartX + barWidth, yloc + 18 + barHeight };
::FillRect(hdc, &bgRect, emptyBrush);
// 確率に応じた塗り幅
int filled = (int)(prob * barWidth);
if (filled < 0) filled = 0;
if (filled > barWidth) filled = barWidth;
// 塗りグラフ（青）
RECT fgRect{ textStartX, yloc + 18, textStartX + filled, yloc + 18 + barHeight };
::FillRect(hdc, &fgRect, filledBrush);
}
// リソース後始末
DeleteObject(emptyBrush);
DeleteObject(filledBrush);
SelectObject(hdc, oldFont);
DeleteObject(font);
}

// ウィンドウプロシージャ
// - WM_PAINT で描画
// - WM_DESTROY でアプリ終了
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
// メッセージで分岐
switch (msg)
{
case WM_PAINT:
{
// 描画開始（ペイント構造体を用意）
PAINTSTRUCT ps;
HDC hdc = BeginPaint(hWnd, &ps);

// クライアント領域サイズ取得
RECT rc;
GetClientRect(hWnd, &rc);
int winW = rc.right - rc.left;
int winH = rc.bottom - rc.top;

// 全体背景を黒で塗りつぶす
::FillRect(hdc, &rc, (HBRUSH)GetStockObject(BLACK_BRUSH));
// グリッド画像を中央に配置するオフセットを計算
int ox, oy;
CalcCenteredOffset(winW, winH, ox, oy);
// グリッド用フレームバッファが存在する場合のみ描画
if (!g_framebuffer.empty())
{
// DIB 用ヘッダ情報
BITMAPINFO bmi = {};
bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
bmi.bmiHeader.biWidth = g_framebufferWidth;
// 上下反転させるため高さを負値に
bmi.bmiHeader.biHeight = -g_fbHeight;
bmi.bmiHeader.biPlanes = 1;
bmi.bmiHeader.biBitCount = 32;
bmi.bmiHeader.biCompression = BI_RGB;
// フレームバッファをそのまま画面へ転送する
StretchDIBits(
hdc,
ox, oy, g_framebufferWidth, g_fbHeight,     // 描画先領域
0, 0, g_framebufferWidth, g_fbHeight,     // 元画像領域
g_framebuffer.data(),
&bmi,
DIB_RGB_COLORS,
SRCCOPY
);
}
// グリッド画像下部のラベル( GT/Pred/O/X ) を描画
DrawLabels(hdc, ox, oy);
// 下部の進捗バーを描画
DrawProgressBar(hdc, winW, winH);
// 右側の詳細ビュー（拡大画像 + Top-10）を描画
DrawDetailView(hdc, ox, oy, winW, winH);
// 描画終了
EndPaint(hWnd, &ps);
return 0;
}

case WM_DESTROY:
// ウィンドウが閉じられたらアプリを終了する
PostQuitMessage(0);
return 0;
}

// その他のメッセージはデフォルト処理に委ねる
return DefWindowProcW(hWnd, msg, wParam, lParam);
}

// ウィンドウ生成処理
bool InitDisplayWindow(int width, int height, const wchar_t* title)
{
// このアプリケーションインスタンスのハンドルを取得
HINSTANCE hi = GetModuleHandle(nullptr);
// ウィンドウクラス構造体の初期化
WNDCLASSW wc{};
// ウィンドウプロシージャ
wc.lpfnWndProc = WndProc;
// インスタンスハンドル
wc.hInstance = hi;
// カーソル（矢印を使用）
wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
// 背景ブラシ（黒）
wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
// クラス名
wc.lpszClassName = L"CIFAR10ViewerW";
// ウィンドウクラスを登録
RegisterClassW(&wc);
// クライアント領域が width×height になるよう枠を調整
RECT rc = { 0, 0, width, height };
AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
// ウィンドウを生成
g_hWnd = CreateWindowExW(
0,                              // 拡張スタイルなし
L"CIFAR10ViewerW",              // クラス名
title,                          // ウィンドウタイトル
WS_OVERLAPPEDWINDOW,            // ウィンドウスタイル
CW_USEDEFAULT,                  // X 位置
CW_USEDEFAULT,                  // Y 位置
rc.right - rc.left,             // 幅
rc.bottom - rc.top,             // 高さ
nullptr,                        // 親ウィンドウなし
nullptr,                        // メニューハンドルなし
hi,                             // インスタンス
nullptr                         // 追加パラメータなし
);

// ウィンドウを表示
ShowWindow(g_hWnd, SW_SHOW);
// 正常に作成されたとみなす
return true;
}

// グリッド画像・ラベル情報を更新し、再描画を要求する
void UpdateDisplayGridWithLabels(
const std::vector<std::vector<uint8_t>>& images,
const std::vector<int>& gtLabels,
const std::vector<int>& predLabels,
const std::vector<bool>& correctFlags,
int imageWidth,
int imageHeight,
int gridColumns,
int scale
)
{
// 画像サイズを保存する
g_imageWidth = imageWidth;
g_imageHeight = imageHeight;
// グリッド列数を保存する
g_gridColumns = gridColumns;
// 拡大スケールを保存する
g_scale = scale;
// 左側グリッド用フレームバッファを構築
BuildFramebuffer(images);
// ラベル情報をコピーする
g_gtLabels = gtLabels;
g_predLabels = predLabels;
g_correctFlags = correctFlags;
// 再描画要求（WM_PAINT を送る）
InvalidateRect(g_hWnd, nullptr, FALSE);
}

// 進捗バーの進捗値をセットし、再描画を要求する
void SetTrainProgress(float p)
{
// 範囲を [0,1] にクランプ
if (p < 0.0f) p = 0.0f;
if (p > 1.0f) p = 1.0f;
// 進捗値を保存
g_progress = p;
// 再描画要求
InvalidateRect(g_hWnd, nullptr, FALSE);
}

// Win32 のメッセージ（キーボード・再描画など）を処理する
void PumpWindowMessages()
{
// メッセージ構造体
MSG msg;

// メッセージキューからすべてのメッセージを処理
while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
// キーボード入力などの変換
TranslateMessage(&msg);
// ウィンドウプロシージャへディスパッチ
DispatchMessageW(&msg);
}
}

// 右側詳細ビューのデータを更新し、再描画を要求する
// main.cpp から呼び出される
void UpdateDetailView(
const std::vector<uint8_t>& image,
const std::vector<std::pair<int, float>>& top10,
const std::vector<std::wstring>& top10Names)
{
// 拡大表示する画像をコピー
g_detailImage = image;
// 詳細画像のサイズ
g_detailImgW = g_imageWidth;
g_detailImgH = g_imageHeight;
// Top-10 のクラスID & 確率をコピー
g_top10 = top10;
// Top-10 に対応するクラス名をコピー
g_top10Names = top10Names;
// 再描画を要求（WM_PAINT を発生させる）
InvalidateRect(g_hWnd, nullptr, FALSE);
}
