import java.util.ArrayList;

import ij.plugin.PlugIn;

public class renderResults_ implements PlugIn {
	public void run(String arg0){
	ArrayList<Particle> correctedResults = TableIO.Load();	
	int inputPixelSize 		= 100;
	int DesiredPixelSize 	= 5;
	int Width 				= 0;
	int Height 				= 0;
	int Pad					= 100;
	for (int i = 0; i < correctedResults.size();i++){
		if (Math.round(correctedResults.get(i).x) > Width){
			Width = (int) Math.round(correctedResults.get(i).x) + Pad;
		}
		if (Math.round(correctedResults.get(i).y) > Height){
			Height = (int) Math.round(correctedResults.get(i).y) + Pad;
		}
	}		
	generateImage.create("test2",correctedResults, Width, Height, DesiredPixelSize);		
	}
}
