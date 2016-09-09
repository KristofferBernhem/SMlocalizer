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
import ij.process.ByteProcessor;
import ij.process.ImageStatistics;

public class generateImage {
	
	public static void create(String Imtitle,ArrayList<Particle> ParticleList, int width, int height, int pixelSize){
		width = Math.round(width/pixelSize);
		height = Math.round(height/pixelSize);
	if (ParticleList.get(ParticleList.size()-1).channel == 1){	
			ByteProcessor IP  = new ByteProcessor(width,height);					
		/*	for (int x = 0; x < width; x++){
				for (int y = 0; y < height; y++){
					IP.putPixel(x, y, 0); // Set all data points to 0 as start.
				}
				
			}*/
			IP.set(0); // set all pixel values to 0 as default.
			
			for (int i = 0; i < ParticleList.size(); i++){
				if (ParticleList.get(i).include == 1){
					int x = (int) Math.round(ParticleList.get(i).x/pixelSize);
					int y = (int) Math.round(ParticleList.get(i).y/pixelSize);				
					IP.putPixel(x, y, (IP.get(x, y) + 1));
				}
			}		
			ImagePlus Image = new ImagePlus(Imtitle,IP);
			Image.setImage(Image);
			Calibration cal = new Calibration(Image);
			cal.pixelHeight = pixelSize;
			cal.pixelWidth 	= pixelSize;
			cal.setXUnit("nm");
			cal.setYUnit("nm");
			ImageStatistics ImStats = Image.getStatistics();
			Image.setDisplayRange(ImStats.min, ImStats.max);
			Image.updateAndDraw();
			Image.setCalibration(cal);
			Image.show(); 														// Make visible
	} // if single channel
	else{
		
		ImageStack imstack = new ImageStack(width,height);
		for (int ch = 1; ch <= ParticleList.get(ParticleList.size()-1).channel; ch ++){
			
		
		ByteProcessor IP  = new ByteProcessor(width,height);					

			IP.set(0); // set all pixel values to 0 as default.
			
			for (int i = 0; i < ParticleList.size(); i++){
				if (ParticleList.get(i).include == 1 && ParticleList.get(i).channel == ch){
					int x = (int) Math.round(ParticleList.get(i).x/pixelSize);
					int y = (int) Math.round(ParticleList.get(i).y/pixelSize);				
					IP.putPixel(x, y, (IP.get(x, y) + 1));
				}
			}
			imstack.addSlice(IP);
		
			
		} // channel loop.
		ImagePlus Image = new ImagePlus(Imtitle, imstack);
		Image.setImage(Image);
		Calibration cal = new Calibration(Image);
		cal.pixelHeight = pixelSize;
		cal.pixelWidth 	= pixelSize;
		cal.setXUnit("nm");
		cal.setYUnit("nm");
		ImageStatistics ImStats = Image.getStatistics();
		Image.setDisplayRange(ImStats.min, ImStats.max);
		Image.updateAndDraw();
		Image.setCalibration(cal);
		Image.show(); 	
	}
	}	
	public static int getIdx(double x, double y, int width, int height){
		int Idx = (int) ( ((y+1)*height) - (width-(x+1)));		
		return Idx;
	}
}

