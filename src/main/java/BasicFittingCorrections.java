/* Copyright 2017 Kristoffer Bernhem.
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
import java.util.ArrayList;

import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageProcessor;

public class BasicFittingCorrections {

	public static ArrayList<Particle> compensate(ArrayList<Particle> result)
	{		 
		if (ij.Prefs.get("SMLocalizer.calibration.2D.channels",0) >= result.get(result.size()-1).channel) // verify that calibration data exists for all channels.
		{
			try{


				double[][] offset = getOffset();
				for (int i = 0; i < result.size(); i++)
				{
					if(result.get(i).channel > 1)
					{
						result.get(i).x -= offset[0][result.get(i).channel-1];
						result.get(i).y -= offset[1][result.get(i).channel-1];
					}
				}
			}
			catch (ArrayIndexOutOfBoundsException e)
			{

			}
		}
		return result;
	}
	public static double[][] getOffset()
	{
		double[][] offset = new double[2][(int) ij.Prefs.get("SMLocalizer.calibration.2D.channels",0)];
		for (int i = 1; i < offset[0].length; i++)
		{
			offset[0][i-1] = ij.Prefs.get("SMLocalizer.calibration.2D.ChOffsetX"+i,0);
			offset[1][i-1] = ij.Prefs.get("SMLocalizer.calibration.2D.ChOffsetY"+i,0);			
		}

		return offset;
	}
	public static void calibrate(int pixelSize, int[] totalGain)
	{
		String modalityChoice 	= "2D";
		double maxSigma 		= 2; // 2D 
		int gWindow = 5;
		if (pixelSize < 100) // if smaller pixel size the window width needs to increase.
		{
			gWindow = (int) Math.ceil(500 / pixelSize); // 500 nm wide window.

		}
		if (gWindow%2 == 0)
			gWindow++;			 
		int  selectedModel = 0;	

		ImagePlus image 					= WindowManager.getCurrentImage();
		int nChannels = image.getNChannels();
		if (nChannels > 1){
			int nFrames 						= image.getNFrames();
			if (nFrames == 1)
				nFrames 						= image.getNSlices(); 
			int[] signalStrength = new int[nChannels];
			for (int ch = 1; ch <= nChannels; ch++)
			{
				image.setPosition(							
						ch,			// channel.
						1,			// slice.
						1);		// frame.
				ImageProcessor IP = image.getProcessor();
				signalStrength[ch-1] = (int) (IP.getMax()*0.9);
			}
			localizeAndFit.run(signalStrength, gWindow, pixelSize,  totalGain , selectedModel, maxSigma, modalityChoice);  //locate and fit all particles.
			ArrayList<Particle> result = TableIO.Load();

			double[][] offset = new double[2][nChannels-1];

			int[] counter = new int[nChannels-1];
			for (int i = 0; i < result.size(); i++)
			{
				if (result.get(i).channel == 1) // first channel.
				{

					int ch = 2;
					while (ch <= nChannels) // reference all subsequent channels against the first one.
					{
						double particleDistance = pixelSize*pixelSize;
						int nearestNeighbor = 0;
						for (int j = i+1; j < result.size(); j++)
						{
							if (result.get(j).channel == ch)
							{
								if (result.get(i).x - result.get(j).x < pixelSize &&
										result.get(i).y - result.get(j).y < pixelSize)
								{
									double tempDist = Math.sqrt((result.get(i).x - result.get(j).x)*(result.get(i).x - result.get(j).x) + 
											(result.get(i).y - result.get(j).y)*(result.get(i).y - result.get(j).y));
									if(tempDist < particleDistance)
									{
										nearestNeighbor = j;
										particleDistance = tempDist;
									}
								}
							}
						}
						counter[ch-2]++;
						offset[0][ch-2] += (result.get(nearestNeighbor).x - result.get(i).x); // remove this offset from channel ch.
						offset[1][ch-2] += (result.get(nearestNeighbor).y - result.get(i).y); // remove this offset from channel ch.

						ch++;
					}

				}
			}
			for(int i = 0; i < nChannels-1; i ++)
			{
				offset[0][i] /=counter[i];
				offset[1][i] /=counter[i];			
			}
			for (int i = 1; i < nChannels; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.2D.ChOffsetX"+i,offset[0][i-1]);
				ij.Prefs.set("SMLocalizer.calibration.2D.ChOffsetY"+i,offset[1][i-1]);
			}
			ij.Prefs.set("SMLocalizer.calibration.2D.channels",nChannels);
			ij.Prefs.savePreferences(); // store settings.
		}else
		{
			for (int i = 1; i < 10; i++)
			{
				ij.Prefs.set("SMLocalizer.calibration.2D.ChOffsetX"+i,0);
				ij.Prefs.set("SMLocalizer.calibration.2D.ChOffsetY"+i,0);
			}
			ij.Prefs.set("SMLocalizer.calibration.2D.channels",0);
			ij.Prefs.savePreferences(); // store settings.
		}

	}

}
