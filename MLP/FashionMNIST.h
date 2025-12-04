#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

// FashionMNISTクラス
//   - IDX フォーマットの Fashion-MNIST データを読み込む構造体
//   - train / test の画像およびラベルを保持
class FashionMNIST
{
public:
	// IDX形式の画像＋ラベルファイルを読み込む
	//  ・imagePath : "train-images-idx3-ubyte"
	//  ・labelPath : "train-labels-idx1-ubyte"
	//  ・isTraining = true  → trainImages / trainLabels に格納する
	//  ・false → testImages  / testLabels に格納する
	bool Load(const std::string& imagePath, const std::string& labelPath, bool isTraining)
	{
		// 画像ファイルをバイナリで開く
		std::ifstream ifsImages(imagePath, std::ios::binary);
		// ラベルファイルをバイナリで開く
		std::ifstream ifsLabels(labelPath, std::ios::binary);
		// 開くことができないエラー復帰する
		if (!ifsImages || !ifsLabels) return false;
		// IDXヘッダの整数をBig-endianで読み込む
		auto ReadInt = [](std::ifstream& stream)
			{
				// 4バイト配列を確保する
				unsigned char headerBytes[4];
				// 4バイト読み込む
				stream.read((char*)headerBytes, 4);
				// Big-endian → int へ変換する
				return (headerBytes[0] << 24) | (headerBytes[1] << 16) | (headerBytes[2] << 8) | headerBytes[3];
			};
		// 画像ファイルヘッダ読み込み（IDX3）
		// マジック番号（2051）を取得する
		int magicImage = ReadInt(ifsImages);  
		// 画像数を取得する
		int numImage = ReadInt(ifsImages);
		// 行数(通常28)を取得する
		int rows = ReadInt(ifsImages);
		// 列数(通常28)を取得する
		int colums = ReadInt(ifsImages);

		// ラベルファイルヘッダ読み込み（IDX1）
		// マジックラベル(2049)を取得する
		int magicLabel = ReadInt(ifsLabels);
		// ラベル数を取得する
		int numLabel = ReadInt(ifsLabels);
		// 画像数とラベル数が一致しない場合はエラー復帰する
		if (numImage != numLabel) return false;
		// 全画像を順に読み込む
		for (int i = 0; i < numImage; i++)
		{
			// 画像(28×28=784 byte)を確保する
			std::vector<uint8_t> images(rows * colums);
			// 画像データ読み込む
			ifsImages.read((char*)images.data(), rows * colums);
			// ラベルを初期化する
			uint8_t label = 0;
			// ラベル(1byte)を読み込む
			ifsLabels.read((char*)&label, 1);

			// 学習データとテストデータとして保存する
			if (isTraining) {
				// 学習データにイメージを保存する
				trainImages.push_back(images);
				// 教師データにラベルを保存する
				trainLabels.push_back(label);
			}
			else {
				// テストイメージにイメージを保存する
				testImages.push_back(images);
				// テストラベルにラベルを保存する
				testLabels.push_back(label);
			}
		}
		// 全て読み込むことができればtrueを返す
		return true;
	}

public:
	// 学習画像配列(28×28=784 byte)
	std::vector<std::vector<uint8_t>> trainImages;
	// 学習ラベル(0〜9)
	std::vector<uint8_t> trainLabels;
	// テスト画像配列
	std::vector<std::vector<uint8_t>> testImages;
	// テストラベル配列
	std::vector<uint8_t> testLabels;
};

