#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>

class Tensor3D
{
public:
	Tensor3D(int h = 0, int w = 0, int c = 0)
		: H(h), W(w), C(c), data(h* w* c, 0.0f)
	{
	}

	void Zero()
	{
		std::fill(data.begin(), data.end(), 0.0f);
	}

	float& operator()(int h, int w, int c)
	{
		if (h < 0 || h >= H ||
			w < 0 || w >= W ||
			c < 0 || c >= C)
		{
			throw std::out_of_range("Tensor3D index OOB");
		}

		return data[(h * W + w) * C + c];
	}

	float operator()(int h, int w, int c) const
	{
		if (h < 0 || h >= H ||
			w < 0 || w >= W ||
			c < 0 || c >= C)
		{
			throw std::out_of_range("Tensor3D index OOB");
		}

		return data[(h * W + w) * C + c];
	}

	int GetH() const { return H; }
	int GetW() const { return W; }
	int GetC() const { return C; }

private:
	int H, W, C;
	std::vector<float> data;
};
