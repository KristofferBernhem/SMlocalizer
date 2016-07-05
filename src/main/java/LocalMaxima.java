
import java.util.ArrayList;

/*
 * 
 */
public class LocalMaxima {
	public static ArrayList<int[]> FindMaxima(float[][] Array, int Window, double MinLevel,double sqDistance, int minPosPixels){		
		ArrayList<int[]> Results = new ArrayList<int[]>();
		//int[] XY = {5,3}; //Example of how results are organized.		
		//Results.add(XY);
	
		int Border = (Window-1)/2;
		for (int i = Border; i < Array.length-Border;i++){ // Look through all columns except outer 2.
			for (int j = Border; j < Array[0].length-Border; j++){ // Look through all rows except outer 2.
				if (Array[i][j] >= MinLevel){ // if center pixel is above user threshold.
					int posPix = 0;
					for (int W = i-Border;W < i+Border; W++){
						for (int H = j-Border; H <j + Border; H++){
							if(Array[W][H]>0){
								posPix++;
							}
							if(Array[W][H]>Array[i][j]){ // If the center pixel is not the strongest in the ROI.
								posPix = -100;
							}
						}										
					}
					if(posPix>=minPosPixels){
						int[] coord = {i,j};					
						Results.add(coord);	
					}

				}

			}
		}

		Results = cleanList(Results,sqDistance);	

		return Results;
	}

	/*
	 * identify overlaps.
	 */
	public static ArrayList<int[]> cleanList(ArrayList<int[]> Array, double Dist){
		int[] Start = {0,0};
		int[] Compare = {0,0}; 
		ArrayList<int[]> CheckedArray = new ArrayList<int[]>();
		int[] found = new int[Array.size()];
		for (int i = 0; i < Array.size(); i++){
			found[i] = 0;
		}
		for (int i = 0; i < Array.size(); i++){

			Start = Array.get(i);
			for (int Check = i+1; Check < Array.size(); Check++){				
				Compare = Array.get(Check);
				double sqDist = ((Start[0] - Compare[0])*(Start[0] - Compare[0]) + 
						(Start[1] - Compare[1])*(Start[1] - Compare[1]));				
				if(sqDist < Dist){
					found[i]++;
					found[Check]++;

				}

				/*
				if (Math.abs(Start[0] - Compare[0]) < Dist &&
						Math.abs(Start[1] - Compare[1]) < Dist){
					int[] entry = {Check};
					Ignore.add(entry);

				}*/
			}
			if (found[i] == 0){				
				int[] coord = Start;
				CheckedArray.add(coord);
			}
			/*}
		if( !Ignore.isEmpty()){
			for (int j = Ignore.size()-1; j >= 0; j--){
				int[] entry = Ignore.get(j);
				Array.remove(entry[0]);
			}
			Array.remove(i);
			Ignore.clear();
		}*/


		}
		return CheckedArray; 
	}
}
