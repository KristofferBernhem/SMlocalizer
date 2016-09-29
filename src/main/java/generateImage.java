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
import java.util.ArrayList;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.process.ImageStatistics;
import ij.process.ShortProcessor;

public class generateImage {
	
	public static void create(String Imtitle,boolean[] renderCh, ArrayList<Particle> ParticleList, int width, int height, int[] pixelSize, boolean gSmoothing){

	if (ParticleList.get(ParticleList.size()-1).channel == 1 && renderCh[0]){
		width = Math.round(width/pixelSize[0]) + 1;
		height = Math.round(height/pixelSize[0] + 1);
		
		ShortProcessor IP  = new ShortProcessor(width,height);					
		IP.set(0); // set all pixel values to 0 as default.
		
		for (int i = 0; i < ParticleList.size(); i++){
			if (ParticleList.get(i).include == 1){
				int x = (int) Math.round(ParticleList.get(i).x/pixelSize[0]);
				int y = (int) Math.round(ParticleList.get(i).y/pixelSize[0]);						
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
		boolean samePixelSize = true;
		for (int ch = 1; ch < ParticleList.get(ParticleList.size()-1).channel; ch++)
		{
			if (pixelSize[ch] != pixelSize[ch-1] && samePixelSize){
				samePixelSize = false;
			}
			
		} // check if all images require the same pixel size.
		
		
		if (samePixelSize) // if user wants an image with the same pixel size, create a stack. Otherwise, create seperate images.
		{
			width = Math.round(width/pixelSize[0]) + 1;
			height = Math.round(height/pixelSize[0] + 1);
		
			ImageStack imstack = new ImageStack(width,height);
		
			for (int ch = 1; ch <= ParticleList.get(ParticleList.size()-1).channel; ch ++){
				if (renderCh[ch-1])
				{
					ShortProcessor IP  = new ShortProcessor(width,height);					
	//				ByteProcessor IP = new ByteProcessor(width,height);
					IP.set(0); // set all pixel values to 0 as default.
					
					for (int i = 0; i < ParticleList.size(); i++){
						if (ParticleList.get(i).include == 1 && ParticleList.get(i).channel == ch){
							int x = (int) Math.round(ParticleList.get(i).x/pixelSize[ch-1]);
							int y = (int) Math.round(ParticleList.get(i).y/pixelSize[ch-1]);				
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
					8);
			Image.setStack(imstack);
			Calibration cal = new Calibration(Image);
			cal.pixelHeight = pixelSize[0];// TODO: check if possible to add calibartion to each slize. 
			cal.pixelWidth 	= pixelSize[0];
			cal.setXUnit("nm");
			cal.setYUnit("nm");
			
			ImageStatistics ImStats = Image.getStatistics();
			
			Image.setDisplayRange(ImStats.min, ImStats.max);
			Image.updateAndDraw();
			Image.setCalibration(cal);
			
			
			Image.show();
		}else
		{
			for(int ch=1; ch <= ParticleList.get(ParticleList.size()-1).channel; ch++)
			{
				if (renderCh[ch-1])
				{
					int widthCh = Math.round(width/pixelSize[ch-1]) + 1;
					int heightCh = Math.round(height/pixelSize[ch-1] + 1);
					
					ShortProcessor IP  = new ShortProcessor(widthCh,heightCh);					
					IP.set(0); // set all pixel values to 0 as default.
					
					for (int i = 0; i < ParticleList.size(); i++){
						if (ParticleList.get(i).include == 1 && 
								ParticleList.get(i).channel == ch){
							int x = (int) Math.round(ParticleList.get(i).x/pixelSize[ch-1]);
							int y = (int) Math.round(ParticleList.get(i).y/pixelSize[ch-1]);						
							IP.putPixel(x, y, (IP.get(x, y) + 1));
						}
					}	
					if(gSmoothing)
					{
						IP.multiply(1000);
						IP.blurGaussian(2);
					}
					ImagePlus Image = new ImagePlus((Imtitle + " channel " + ch),IP);
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
					Image.show();
				}// check if the channel should be rendered.
			}// loop over channel
		}
	}
	}	
	public static int getIdx(double x, double y, int width, int height){
		int Idx = (int) ( ((y+1)*height) - (width-(x+1)));		
		return Idx;
	}
}

