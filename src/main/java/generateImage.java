package sm_localizer;

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
			int x = (int) Math.round(ParticleList.get(i).x/pixelSize);
			int y = (int) Math.round(ParticleList.get(i).y/pixelSize);		
			IP.putPixel(x, y, 255);
			
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
