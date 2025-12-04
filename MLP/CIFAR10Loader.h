#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

// CIFAR10Loaderクラス
//   - CIFAR-10 バイナリフォーマットのデータを読み込む構造体
//   - train / test の画像およびラベルを保持
//   - 画像サイズ: 32×32 RGB (3チャンネル)
//   - 10クラス分類
class CIFAR10Loader
{
public:
	// CIFAR-10 のクラス名（10クラス）
	static const wchar_t* GetClassName(int classId)
	{
		static const wchar_t* names[10] =
		{
			L"airplane",    // 0: 飛行機
			L"automobile",  // 1: 自動車
			L"bird",        // 2: 鳥
			L"cat",         // 3: 猫
			L"deer",        // 4: 鹿
			L"dog",         // 5: 犬
			L"frog",        // 6: カエル
			L"horse",       // 7: 馬
			L"ship",        // 8: 船
			L"truck"        // 9: トラック
		};
		if (classId < 0 || classId >= 10) return L"unknown";
		return names[classId];
	}

	// CIFAR-10バイナリファイルを読み込む
	// - binPath : "cifar-10-batches-bin/data_batch_1.bin" など
	// - isTraining = true  → trainImages / trainLabels に格納する
	// - isTraining = false → testImages / testLabels に格納する
	bool LoadBatch(const std::string& binPath, bool isTraining)
	{
		// バイナリファイルをバイナリモードで開く
		std::ifstream ifs(binPath, std::ios::binary);
		// 開くことができないならエラー復帰する
		if (!ifs) return false;

		// CIFAR-10の各バッチは10000枚の画像を含む
		// 各画像レコード: 1バイト(ラベル) + 3072バイト(32×32×3 RGB)
		const int recordSize = 1 + 32 * 32 * 3;
		const int numImages = 10000;

		// 全画像を順に読み込む
		for (int i = 0; i < numImages; i++)
		{
			// ラベル(1バイト)を読み込む
			uint8_t label = 0;
			ifs.read(reinterpret_cast<char*>(&label), 1);

			// EOF チェック
			if (ifs.eof()) break;

			// 画像データ(3072バイト = 32×32×3)を読み込む
			// CIFAR-10は平面順: R(1024) + G(1024) + B(1024)
			std::vector<uint8_t> rawData(32 * 32 * 3);
			ifs.read(reinterpret_cast<char*>(rawData.data()), 32 * 32 * 3);

			// RGB平面順からインターリーブ形式(RGBRGBRGB...)に変換
			// 表示用にピクセル順に並び替える
			std::vector<uint8_t> imageData(32 * 32 * 3);
			for (int y = 0; y < 32; y++)
			{
				for (int x = 0; x < 32; x++)
				{
					int pixelIndex = y * 32 + x;
					// R チャンネル (最初の1024バイト)
					imageData[pixelIndex * 3 + 0] = rawData[0 * 1024 + pixelIndex];
					// G チャンネル (次の1024バイト)
					imageData[pixelIndex * 3 + 1] = rawData[1 * 1024 + pixelIndex];
					// B チャンネル (最後の1024バイト)
					imageData[pixelIndex * 3 + 2] = rawData[2 * 1024 + pixelIndex];
				}
			}

			// 学習データまたはテストデータとして保存する
			if (isTraining) {
				// 学習データにイメージを保存する
				trainImages.push_back(imageData);
				// 教師データにラベルを保存する
				trainLabels.push_back(label);
			}
			else {
				// テストイメージにイメージを保存する
				testImages.push_back(imageData);
				// テストラベルにラベルを保存する
				testLabels.push_back(label);
			}
		}
		// 全て読み込むことができればtrueを返す
		return true;
	}

	// 訓練データをすべて読み込む（5つのバッチファイル）
	// basePath: "cifar-10-batches-bin/" など
	bool LoadAllTrainData(const std::string& basePath)
	{
		bool success = true;
		// 5つの訓練バッチを順に読み込む
		for (int i = 1; i <= 5; i++)
		{
			std::string path = basePath + "data_batch_" + std::to_string(i) + ".bin";
			if (!LoadBatch(path, true))
			{
				success = false;
			}
		}
		return success;
	}

	// テストデータを読み込む
	// basePath: "cifar-10-batches-bin/" など
	bool LoadTestData(const std::string& basePath)
	{
		std::string path = basePath + "test_batch.bin";
		return LoadBatch(path, false);
	}

public:
	// 学習画像配列(32×32×3 = 3072 byte、RGB インターリーブ形式)
	std::vector<std::vector<uint8_t>> trainImages;
	// 学習ラベル(0〜9)
	std::vector<uint8_t> trainLabels;
	// テスト画像配列
	std::vector<std::vector<uint8_t>> testImages;
	// テストラベル配列
	std::vector<uint8_t> testLabels;

	// 画像の定数
	static constexpr int IMAGE_WIDTH = 32;
	static constexpr int IMAGE_HEIGHT = 32;
	static constexpr int IMAGE_CHANNELS = 3;
	static constexpr int NUM_CLASSES = 10;
};
