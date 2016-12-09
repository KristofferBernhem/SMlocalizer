/* Copyright 2016 Kristoffer Bernhem.
 * This file is part of SMLocalizer.
 *
 *  SMLocalizer is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SMLocalizer is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SMLocalizer.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */
import java.util.Random;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

public class TranslateIm {

	public static int[][][][] ReadIm() // read current image and translate it into a 4D stack.
	{
		ImagePlus image = WindowManager.getCurrentImage();
		int nChannels 	= image.getNChannels();
		int nFrames 	= image.getNFrames();
		if (nFrames == 1)
			nFrames = image.getNSlices();  				// some formats store frames as slices, some as frames.
		int rows 		= image.getWidth();
		int columns 	= image.getHeight();
		int[][][][] outputData = new int[rows][columns][nFrames][nChannels];
		
		if (nChannels == 1)
		{
			for (int Frame = 1; Frame <= nFrames; Frame++){			
				image.setSlice(Frame);
				ImageProcessor IP = image.getProcessor();
				for (int i = 0; i < rows; i++){
					for (int j = 0; j < columns; j++){
						outputData[i][j][Frame-1][0] = IP.getPixel(i, j);
					}
				}	
				
			}
		} // single channel
		else // multichannel imagestack 
		{ 
			for (int Ch = 1; Ch <= nChannels; Ch++){ // Loop over all channels.				
				ImageProcessor IP = image.getProcessor();		// get image processor for the stack.
					for (int Frame = 1; Frame < nFrames+1; Frame++){			
					image.setPosition(
							Ch,			// channel.
							1,			// slice.
							Frame);		// frame.
					IP 						= image.getProcessor(); 			// Update processor to next slice.
					
					for (int i = 0; i < rows; i++){
						for (int j = 0; j < columns; j++){
							outputData[i][j][Frame-1][Ch-1] = IP.getPixel(i, j);
						}
					} 										
				} // loop over frames.
			} // loop over channels
		} // multichannel imagestack.
		
		return outputData;
	}
	
	public static void MakeIm(int[][][][] inputData)
	{
		int nChannels 	= inputData[0][0][0].length;
		int nFrames 	= inputData[0][0].length;
		int rows 		= inputData.length;
		int columns 	= inputData[0].length;
		ImagePlus IM = ij.IJ.createHyperStack("", rows, columns, nChannels, 1, nFrames, 16);
		ImageStack imStack = new ImageStack(rows, columns);
			for (int frame = 1; frame <= nFrames; frame++){
				for (int ch = 1; ch <= nChannels; ch++){
			
				ShortProcessor IP = new ShortProcessor(rows, columns);
				int[][] imageSlize = new int[rows][columns];
				for (int x = 0; x < rows; x++)
				{
					for (int y = 0; y < columns; y++)
					{
						imageSlize[x][y] = inputData[x][y][frame-1][ch-1];
					}
				}
				IP.setIntArray(imageSlize);
				imStack.addSlice(IP);
			}
		}
		IM.setStack(imStack);
		IM.show();
		
	}
	public static void main(String[] args) {
		int xM = 64;
		int yM = 64;
		int fM = 1000;
		int cM = 3;
		int[][][][] testData = new int[xM][yM][fM][cM];
		Random r = new Random();
		for (int channel = 0; channel < cM; channel ++)
		{
			for (int frame = 0; frame < fM; frame++ )
			{
				for (int x = 0; x < xM; x++)
				{
					for (int y = 0; y < yM; y++)
					{
						if (channel == 0)
							testData[x][y][frame][channel] = r.nextInt(2^16-1);
						else
							testData[x][y][frame][channel] = 0;
					}
				}
			}
		}
		
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);

		
		 MakeIm(testData);
	}

}
