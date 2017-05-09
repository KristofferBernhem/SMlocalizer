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


import ij.ImagePlus;
import ij.process.ImageProcessor;

public class findLimits {

	public static int[] run(ImagePlus image, int ch)
	{
		int columns 						= image.getWidth();
		int rows 							= image.getHeight();		
		int nFrames 						= image.getNFrames();
		if (nFrames == 1)
			nFrames 						= image.getNSlices();

		int[] limits = new int[nFrames];

		double[] runningMean = new double[nFrames];
		double[] runningStd = new double[nFrames];
		int meanWidth = 500;
		if (meanWidth > nFrames)
			meanWidth = nFrames;
		int count = 0;
		for (int frame  = 1; frame <= nFrames; frame++)
		{

			count = 0;
			if (image.getNFrames() == 1)
			{
				image.setPosition(							
						ch,			// channel.
						frame,			// slice.
						1);		// frame.
			}
			else
			{														
				image.setPosition(
						ch,			// channel.
						1,			// slice.
						frame);		// frame.
			}
			ImageProcessor IP = image.getProcessor();
			// get mean for the frame.
			for (int id = 0; id < columns*rows; id++){
				runningMean [frame-1] += IP.get(id);
				if (IP.get(id) > 0)
					count++;
			}

			runningMean[frame-1]/=count;
			// get std for the frame.
			for (int id = 0; id < columns*rows;id++)
				runningStd[frame-1] += (IP.get(id) - runningMean[frame-1])*(IP.get(id) - runningMean[frame-1]);

			runningStd[frame-1] /= count;
			runningStd[frame-1] = Math.sqrt(runningStd[frame-1]);
		}
		for (int i = 0; i < limits.length; i++)
		{
			double meanOfMeans 	= 0;
			double meanOfStd 	= 0;
			int idx 			= i - meanWidth-1;
			count = 0;
			if (idx < 0)
				idx = 0;
			while (idx < runningMean.length && idx < i + meanWidth)
			{
				meanOfMeans += runningMean[idx];
				meanOfStd += runningStd[idx];
				count++;
				idx++;
			}
			meanOfMeans /= count;
			meanOfStd /= count;
			limits[i] = (int) (meanOfMeans*2 + meanOfStd*4);

		}		

		return limits;
	}
	public static int[] run(int[] image, int columns, int rows, int ch)
	{
		int nFrames = (int)(image.length/(columns*rows));
		int[] limits = new int[nFrames];

		double[] runningMean = new double[nFrames];
		double[] runningStd = new double[nFrames];
		int meanWidth = 500;
		if (meanWidth > nFrames)
			meanWidth = nFrames;
		int count = 0;

		for (int frame  = 1; frame <= nFrames; frame++)
		{

			count = 0;
			int frameOffset = (frame-1)*columns*rows;
			for (int id = 0; id < columns*rows; id++){				
				runningMean [frame-1] += image[id+frameOffset];
				if (image[id + frameOffset] > 0)
					count++;
				
			}

			runningMean[frame-1]/=count;
			// get std for the frame.
			for (int id = 0; id < columns*rows;id++)
				runningStd[frame-1] += (image[id+frameOffset] - runningMean[frame-1])*(image[id+frameOffset] - runningMean[frame-1]);

			runningStd[frame-1] /= count;
			runningStd[frame-1] = Math.sqrt(runningStd[frame-1]);
		}
		for (int i = 0; i < limits.length; i++)
		{
			double meanOfMeans 	= 0;
			double meanOfStd 	= 0;
			int idx 			= i - meanWidth-1;
			count = 0;
			if (idx < 0)
				idx = 0;
			while (idx < runningMean.length && idx < i + meanWidth)
			{
				meanOfMeans += runningMean[idx];
				meanOfStd += runningStd[idx];
				count++;
				idx++;
			}
			meanOfMeans /= count;
			meanOfStd /= count;
			limits[i] = (int) (meanOfMeans*2 + meanOfStd*4);

		}		

		return limits;
	}
}
