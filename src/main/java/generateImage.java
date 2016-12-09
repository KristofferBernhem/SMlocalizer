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

/*
 * TODO: 3D rendering!
 */
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */
import java.util.ArrayList;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.process.ImageStatistics;
import ij.process.ShortProcessor;

public class generateImage {

	public static void create(String Imtitle,boolean[] renderCh, ArrayList<Particle> ParticleList, int width, int height, int[] pixelSize, boolean gSmoothing){

		boolean threeD = false;
		int idx = 0;
		while( idx < ParticleList.size() && !threeD)
		{
			if (ParticleList.get(idx).z > 0)
				threeD = true;
			idx++;
		}
		if (threeD)
		{
			width = (int) Math.ceil(width/pixelSize[0]);
			height = (int) Math.ceil(height/pixelSize[0]);		
			ImageStack imstack = new ImageStack(width,height);
			int nChannels = ParticleList.get(ParticleList.size()-1).channel; // number of channels
			int zSlice = 0; // counter for number of slices.
			int particleCounter = 0;
			/*
			 * Loop over all particles, add the ones that has the correct channel and z-range. Increase channel until all channels are done then update z-range
			 */
			int Ch = 1;
			while (particleCounter < ParticleList.size())
			{
				ShortProcessor IP  = new ShortProcessor(width,height);					
				IP.set(0); // set all pixel values to 0 as default.
				int lowZ = zSlice*pixelSize[1]; // low limit
				int highZ = (zSlice+1)*pixelSize[1]; // high limit
				for (idx = 0; idx < ParticleList.size(); idx++)
				{
					if (ParticleList.get(idx).channel == Ch &&
							ParticleList.get(idx).z >= lowZ &&
							ParticleList.get(idx).z < highZ)
					{
						int x = (int) Math.round(ParticleList.get(idx).x/pixelSize[0]);
						int y = (int) Math.round(ParticleList.get(idx).y/pixelSize[0]);						
						if (x >= 0 && x <= width &&
								y >= 0 && y <= height)
							IP.putPixel(x, y, (IP.get(x, y) + 1));
					}
				}
				if(gSmoothing)
				{
					IP.multiply(1000);
					IP.blurGaussian(2);
				}
				Ch++;
				if (Ch > nChannels)
				{
					Ch = 1;
					zSlice++;
				}
				particleCounter++; // keep track of number of added particles.
				imstack.addSlice(IP);
			}

			ImagePlus Image = ij.IJ.createHyperStack(Imtitle, 
					width, 
					height, 
					nChannels, 
					zSlice, 
					1, 
					16);
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
		} // 3D images end
		else // if user wants to render 2D data.
		{
			if (ParticleList.get(ParticleList.size()-1).channel == 1 && renderCh[0]){
				width = (int) Math.ceil(width/pixelSize[0]);
				height = (int) Math.ceil(height/pixelSize[0]);		
				ShortProcessor IP  = new ShortProcessor(width,height);					
				IP.set(0); // set all pixel values to 0 as default.

				for (int i = 0; i < ParticleList.size(); i++){
					if (ParticleList.get(i).include == 1){
						int x = (int) Math.round(ParticleList.get(i).x/pixelSize[0]);
						int y = (int) Math.round(ParticleList.get(i).y/pixelSize[0]);
						if (x >= 0 && x <= width &&
								y >= 0 && y <= height)
							IP.putPixel(x, y, (IP.get(x, y) + 1));
					}
				}	
				if(gSmoothing)
				{
					IP.multiply(1000);
					IP.blurGaussian(2);
				}
				ImagePlus Image = new ImagePlus(Imtitle,IP);
				Image.setImage(Image);
				Calibration cal = new Calibration(Image);
				cal.pixelHeight = pixelSize[0];
				cal.pixelWidth 	= pixelSize[0];
				cal.setXUnit("nm");
				cal.setYUnit("nm");
				ImageStatistics ImStats = Image.getStatistics();
				Image.setDisplayRange(ImStats.min, ImStats.max);
				Image.updateAndDraw();
				Image.setCalibration(cal);
				Image.show(); 														// Make visible
			} // if single channel
			else{			
				width = (int) Math.ceil(width/pixelSize[0]);
				height = (int) Math.ceil(height/pixelSize[0]);	

				ImageStack imstack = new ImageStack(width,height);

				for (int ch = 1; ch <= ParticleList.get(ParticleList.size()-1).channel; ch ++){
					if (renderCh[ch-1])
					{
						ShortProcessor IP  = new ShortProcessor(width,height);					
						IP.set(0); // set all pixel values to 0 as default.

						for (int i = 0; i < ParticleList.size(); i++){
							if (ParticleList.get(i).include == 1 && ParticleList.get(i).channel == ch){
								int x = (int) Math.round(ParticleList.get(i).x/pixelSize[0]);
								int y = (int) Math.round(ParticleList.get(i).y/pixelSize[0]);				
								if (x >= 0 && x <= width &&
										y >= 0 && y <= height)
									IP.putPixel(x, y, (IP.get(x, y) + 1));
							}
						}
						if(gSmoothing)
						{
							IP.multiply(1000);
							IP.blurGaussian(2);
						}
						imstack.addSlice(IP);
					} // if the channel should be drawn.

				} // channel loop.

				ImagePlus Image = ij.IJ.createHyperStack(Imtitle, 
						width, 
						height, 
						(int) ParticleList.get(ParticleList.size()-1).channel, 
						1, 
						1, 
						16);
				Image.setStack(imstack);
				Calibration cal = new Calibration(Image);
				cal.pixelHeight = pixelSize[0]; 
				cal.pixelWidth 	= pixelSize[0];
				cal.setXUnit("nm");
				cal.setYUnit("nm");

				ImageStatistics ImStats = Image.getStatistics();

				Image.setDisplayRange(ImStats.min, ImStats.max);
				Image.updateAndDraw();
				Image.setCalibration(cal);


				Image.show();
			}// 2D images.
		}
	}	
	public static int getIdx(double x, double y, int width, int height){
		int Idx = (int) ( ((y+1)*height) - (width-(x+1)));		
		return Idx;
	}
}

