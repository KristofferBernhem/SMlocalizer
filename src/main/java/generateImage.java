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
import ij.process.ByteProcessor;

public class generateImage {
	
	public static void create(String Imtitle,ArrayList<Particle> ParticleList, int width, int height, int pixelSize){
		width = Math.round(width/pixelSize);
		height = Math.round(height/pixelSize);
		ByteProcessor IP  = new ByteProcessor(width,height);			
		for (int x = 0; x < width; x++){
			for (int y = 0; y < height; y++){
				IP.putPixel(x, y, 0); // Set all data points to 0 as start.
			}
			
		}
	
		for (int i = 0; i < ParticleList.size(); i++){
			if (ParticleList.get(i).include == 1){
				int x = (int) Math.round(ParticleList.get(i).x/pixelSize);
				int y = (int) Math.round(ParticleList.get(i).y/pixelSize);		
				IP.putPixel(x, y, 1);
			}
		}		
		ImagePlus Image = new ImagePlus(Imtitle,IP);
		Image.setImage(Image);
		Image.show(); 														// Make visible
		
		
	}	
	public static int getIdx(double x, double y, int width, int height){
		int Idx = (int) ( ((y+1)*height) - (width-(x+1)));		
		return Idx;
	}
}
