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
import java.util.ArrayList;

import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.plugin.filter.Analyzer;
import ij.process.ByteProcessor;
import ij.process.ImageStatistics;


public class SMLocalizerMovieRenderer{// implements PlugIn {


	public static void run(int[] pixelSize, int frameBinSize, boolean GaussFilter)
	{
		ArrayList<Particle> ParticleList 	= TableIO.Load(); 									// get results table.
		ij.measure.ResultsTable tab 		= Analyzer.getResultsTable();   					// load a new handle to results table.
		int width 							= (int)tab.getValue("width", 0);					// get width of image.
		int height 							= (int) tab.getValue("height", 0);					// get height of image.	
		width 								= (int) Math.ceil(width/pixelSize[0]);				// rescale image.
		height 								= (int) Math.ceil(height/pixelSize[0]);				// rescale image.
		ImageStack imstack 					= new ImageStack(width,height);						// generate image stack that will become the final output.
		int nChannels 						= ParticleList.get(ParticleList.size()-1).channel; 	// number of channels
		int[] idxChannelStart 				= new int[nChannels+1];								// start index for channels.
		idxChannelStart[0] 					= 0;
		idxChannelStart[nChannels] 			= ParticleList.size();
		int maxFrame						= 1;
		int count = 0;
		for (int i = 0; i < ParticleList.size(); i++)
		{
			if (ParticleList.get(i).frame > maxFrame)
				maxFrame = ParticleList.get(i).frame;
			if (ParticleList.get(i).channel > ParticleList.get(idxChannelStart[count]).channel)
			{
				count++;
				idxChannelStart[count] = i;
			}
		}
		int frame = 1;
		while (frame <= maxFrame)
		{
			for (int channel = 1; channel <= nChannels; channel++)
			{
				ByteProcessor IP  = new ByteProcessor(width,height);	// 8 bit frame.
				IP.set(0);
				for (int idx = idxChannelStart[channel-1]; idx < idxChannelStart[channel]; idx++)
				{
					if (ParticleList.get(idx).frame <= frame)
					{
						int x = (int) Math.round(ParticleList.get(idx).x/pixelSize[0]);
						int y = (int) Math.round(ParticleList.get(idx).y/pixelSize[0]);						
						IP.putPixel(x, y, (IP.get(x, y) + 1));
					} // frame check.
				} // idx for loop.
				if (GaussFilter){ // if user wants gaussian smoothing.
					IP.multiply(50); 			
					IP.blurGaussian(2);			
				}
				imstack.addSlice(IP);
			} // channel loop.
			frame+=frameBinSize;  // Frame step size. Increase to decrease movie size.
		} // main loop over all frames.

		ImagePlus Image = ij.IJ.createHyperStack("", 
				width, 
				height, 
				nChannels, 
				1, 
				maxFrame, 
				8);
		Image.setStack(imstack);
		Calibration cal = new Calibration(Image);
		cal.pixelHeight = pixelSize[0]; 
		cal.pixelWidth 	= pixelSize[0];
		cal.setXUnit("nm");
		cal.setYUnit("nm");

		ImageStatistics ImStats = Image.getStatistics();
		Image.setCalibration(cal);
		Image.setDisplayRange(ImStats.min, ImStats.max);
		Image.updateAndDraw();
		Image.show();

	}
}
