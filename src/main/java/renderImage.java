import java.util.ArrayList;

public class renderImage {
	public static void run(int inputPixelSize, int DesiredPixelSize ){
		try{
			ArrayList<Particle> correctedResults = TableIO.Load();	
			int Width 				= 0;
			int Height 				= 0;
			int Pad					= 100;

			int PixelRatio = Math.round(inputPixelSize/DesiredPixelSize);
			for (int i = 0; i < correctedResults.size();i++){
				if (correctedResults.get(i).include == 1){ 

					if (Math.round(correctedResults.get(i).x) > Width){
						Width = (int) Math.round(correctedResults.get(i).x) + Pad;
					}
					if (Math.round(correctedResults.get(i).y) > Height){
						Height = (int) Math.round(correctedResults.get(i).y) + Pad;
					}
				}
			}		
			generateImage.create("RenderedResults",correctedResults, Width, Height, PixelRatio);		
		}
		catch (Exception e){
		}
	}
}
