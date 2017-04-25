// Copyright 2017 Pok On Cheng
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/***************************
  * Pok On Cheng (pocheng) *
  * 74157306               *
  * CompSci 131 Lab 1 B    *
  ***************************/

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <mutex>
#include <thread>

 
/* Global variables, Look at their usage in main() */
int image_height;
int image_width;
int image_maxShades;
int inputImage[1000][1000];
int outputImage[1000][1000];
int num_threads; 
int chunkSize;
int maxChunk;
int nextAvailableChunk;
std::mutex mutexes;

/* ****************Change and add functions below ***************** */
int get_dynamic_chunk() {
	mutexes.lock();
	int N = chunkSize * nextAvailableChunk;
	nextAvailableChunk += 1;
	mutexes.unlock();
	return N;
}

void sobel_algorithm(int dynamic_chunk) {
	int sumx, sumy, sum;
	int maskX[3][3];
	int maskY[3][3];
	
	/* 3x3 Sobel mask for X Dimension. */
	maskX[0][0] = -1; maskX[0][1] = 0; maskX[0][2] = 1;
	maskX[1][0] = -2; maskX[1][1] = 0; maskX[1][2] = 2;
	maskX[2][0] = -1; maskX[2][1] = 0; maskX[2][2] = 1;
	
	/* 3x3 Sobel mask for Y Dimension. */
	maskY[0][0] = 1; maskY[0][1] = 2; maskY[0][2] = 1;
	maskY[1][0] = 0; maskY[1][1] = 0; maskY[1][2] = 0;
	maskY[2][0] = -1; maskY[2][1] = -2; maskY[2][2] = -1;
	for (int x = dynamic_chunk; x < (dynamic_chunk+chunkSize); ++x) {
		for (int y = 0; y < image_width; ++y){
			sumx = 0;
			sumy = 0;

			/* For handling image boundaries */
			if (x == 0 || x == ((dynamic_chunk+chunkSize)-1) || y == 0 || y == (image_width-1)) {
				sum = 0;
			}
			else {
				/* Gradient calculation in X Dimension */
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++){
						sumx += (inputImage[x+i][y+j] * maskX[i+1][j+1]);
					}
				}

				/* Gradient calculation in Y Dimension */
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++){
						sumy += (inputImage[x+i][y+j] * maskY[i+1][j+1]);
					}
				}

				/* Gradient magnitude */
				sum = (abs(sumx) + abs(sumy));
			}

			/* outputImage[x][y] = (0 <= sum <= 255); */
			if (sum >= 0 && sum <= 255) {
				outputImage[x][y] = sum;
			}
			else if (sum > 255) {
				outputImage[x][y] = 0;
			}
			else if (sum < 0) {
				outputImage[x][y] = 255;
			}
		}
	}
}

void compute_chunk() {
	int X = get_dynamic_chunk();
	sobel_algorithm(X);
}

void dispatch_threads()
{
	std::vector<std::thread> threads;
	nextAvailableChunk = 0;
	
	for (int i = 0; i < num_threads; i++) {
		threads.push_back(std::thread(compute_chunk));
	}
	
	for (int i = 0; i < num_threads; i++) {
		threads[i].join();
	}
}

/* ****************Need not to change the function below ***************** */

int main(int argc, char* argv[])
{
    if(argc != 5)
    {
        std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename> <Threads#> <Chunk size>" << std::endl;
        return 0;
    }
 
    std::ifstream file(argv[1]);
    if(!file.is_open())
    {
        std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
        return 0;
    }
    num_threads = std::atoi(argv[3]);
    chunkSize  = std::atoi(argv[4]);

    std::cout << "Detect edges in " << argv[1] << " using " << num_threads << " threads\n" << std::endl;

    /* ******Reading image into 2-D array below******** */

    std::string workString;
    /* Remove comments '#' and check image format */ 
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            if( workString.at(1) != '2' ){
                std::cout << "Input image is not a valid PGM image" << std::endl;
                return 0;
            } else {
                break;
            }       
        } else {
            continue;
        }
    }
    /* Check image size */ 
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            int n;
            stream >> n;
            image_width = n;
            stream >> n;
            image_height = n;
            break;
        } else {
            continue;
        }
    }

    /* maxChunk is total number of chunks to process */
    maxChunk = ceil((float)image_height/chunkSize);

    /* Check image max shades */ 
    while(std::getline(file,workString))
    {
        if( workString.at(0) != '#' ){
            std::stringstream stream(workString);
            stream >> image_maxShades;
            break;
        } else {
            continue;
        }
    }
    /* Fill input image matrix */ 
    int pixel_val;
    for( int i = 0; i < image_height; i++ )
    {
        if( std::getline(file,workString) && workString.at(0) != '#' ){
            std::stringstream stream(workString);
            for( int j = 0; j < image_width; j++ ){
                if( !stream )
                    break;
                stream >> pixel_val;
                inputImage[i][j] = pixel_val;
            }
        } else {
            continue;
        }
    }

    /************ Function that creates threads and manage dynamic allocation of chunks *********/
    dispatch_threads();

    /* ********Start writing output to your file************ */
    std::ofstream ofile(argv[2]);
    if( ofile.is_open() )
    {
        ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
        for( int i = 0; i < image_height; i++ )
        {
            for( int j = 0; j < image_width; j++ ){
                ofile << outputImage[i][j] << " ";
            }
            ofile << "\n";
        }
    } else {
        std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
        return 0;
    }
    return 0;
}