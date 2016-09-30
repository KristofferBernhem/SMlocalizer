import java.util.ArrayList;

public class nearestNeighbour {

	public static void analyse()
	{
		ArrayList<Particle> results = TableIO.Load();
		int[] startIdx = new int[results.get(results.size()-1).channel];
		int currIdx = 0;
		startIdx[0]  = 0; // first comparion  starts at index 0.
		for (int i = 0; i < results.size(); i++)
		{
			if (results.get(i).channel > results.get(startIdx[currIdx]).channel &&
					results.get(i).channel  > 1)
				{				
				
					currIdx++;					
					startIdx[currIdx] = i;	
						
				}				
		} // obtain which groups to compare with.
		double[] nnResults = new double[25]; // 10 nm steps with final bin being for larger distances.
		int binSize = 10;
		// this loops over ch1 and compares to ch2 only.
		for (int referenceIdx = startIdx[1]; referenceIdx < results.size()-1;referenceIdx++ )
		//for (int referenceIdx = startIdx[0]; referenceIdx < startIdx[1]-1; referenceIdx++)
		{
			double distance = 10000;
			for (int targetIdx = startIdx[0]; targetIdx < startIdx[1]-1; targetIdx++)
			//for (int targetIdx = startIdx[1]; targetIdx < results.size()-1;targetIdx++ )
			{
				double tempDistance = Math.sqrt((results.get(referenceIdx).x-results.get(targetIdx).x)*(results.get(referenceIdx).x-results.get(targetIdx).x) +
						(results.get(referenceIdx).y-results.get(targetIdx).y)*(results.get(referenceIdx).y-results.get(targetIdx).y) + 
						(results.get(referenceIdx).z-results.get(targetIdx).z)*(results.get(referenceIdx).z-results.get(targetIdx).z)
						);
				if (tempDistance < distance)
					distance = tempDistance;
			}
			boolean sorted = false;
			int bin = 1;			
			while (!sorted)
			{
				if (distance < (double)bin*binSize)
					{
						nnResults[bin-1]++;
						sorted = true;
					}
				bin++;
				if (bin == 26) // exit
				{
				//	nnResults[25]++;
					sorted = true;
				}
				
			}
					
		} // nnResults now populated.
		// display nnResults:
		int[] x = {0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250};
		correctDrift.plot(nnResults, x);
		for (int i = 0; i < nnResults.length;i++)
			nnResults[i] /= (startIdx[1] - startIdx[0]);
		correctDrift.plot(nnResults, x);
	}
	
}
