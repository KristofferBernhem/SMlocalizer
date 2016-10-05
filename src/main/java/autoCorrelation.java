import java.util.ArrayList;

import ij.process.ByteProcessor;


public class autoCorrelation {

	
	
	
	public static double correlation(short[][] referenceIP, short[][] targetIP, int[] shift)
	{
		double corr = 0;		
		for(int i = Math.abs(shift[0]); i < referenceIP.length-Math.abs(shift[0]);i++)
		{
			for(int j = Math.abs(shift[1]); j < referenceIP[0].length-Math.abs(shift[1]); j++)
			{
				corr += referenceIP[i][j]*targetIP[i-shift[0]][j-shift[1]];
			}
		}
		return corr;
	}
	public static double correlation(ByteProcessor referenceIP, ByteProcessor targetIP, int[] shift)
	{
		
		double corr = 0;		
		for(int i = Math.abs(shift[0]); i <referenceIP.getWidth()-Math.abs(shift[0]);i++)
		{
			for(int j = Math.abs(shift[1]); j < referenceIP.getHeight()-Math.abs(shift[1]); j++)
			{
				if( referenceIP.get(i, j) != 0)
				{
					corr = corr + referenceIP.get(i, j)*targetIP.get(i-shift[0], j-shift[1]);		
				}
			}
		}
		return corr;
	}
	public static double correlation(ArrayList<Particle> referenceIP, ArrayList<Particle> targetIP, int[] shift)
	{
		
		double corr = 0;		
		for (int i = 0; i < referenceIP.size(); i++)
		{
			for (int j = 0; j < targetIP.size(); j++)
			{
				if (Math.abs(referenceIP.get(i).x - targetIP.get(j).x - shift[0]) < 5 &&
						Math.abs(referenceIP.get(i).y - targetIP.get(j).y - shift[1]) < 5)
				{
					corr++;
				}
			}
		}
		return corr;
	}
	public static double[] minimize(ByteProcessor referenceIP, ByteProcessor targetIP, int[] maxShift)
	{
		int[] shift = {0,0};
		
		double[] corr = {0,0,0};
		for (int xShift = -maxShift[0]; xShift <= maxShift[0]; xShift+=maxShift[0]/10)
		{
			for (int yShift = -maxShift[1]; yShift <= maxShift[1]; yShift += maxShift[1]/10)
			{
				shift[0] = xShift;
				shift[1] = yShift;
				double tempCorr = correlation(referenceIP, targetIP,shift);
				if (tempCorr > corr[0])
				{
					corr[0] = tempCorr;
					corr[1] = shift[0];
					corr[2] = shift[1];
				}
			}	
		}
		return corr;
	}
	public static double[] minimize(short[][] referenceIP, short[][] targetIP, int[] maxShift)
	{
		// TODO: write iterative version for this.
		int[] shift = {0,0};
		
		double[] corr = {0,0,0};
		for (int xShift = -maxShift[0]; xShift <= maxShift[0]; xShift+= 5)//maxShift[0]/10)
		{
			for (int yShift = -maxShift[1]; yShift <= maxShift[1]; yShift += 5)// maxShift[1]/10)
			{
				shift[0] = xShift;
				shift[1] = yShift;
				double tempCorr = correlation(referenceIP, targetIP,shift);
				if (tempCorr > corr[0])
				{
					corr[0] = tempCorr;
					corr[1] = shift[0];
					corr[2] = shift[1];
				}
			}	
		}
		return corr;
	}
	public static double[] minimize(ArrayList<Particle> referenceIP, ArrayList<Particle> targetIP, int[] maxShift)
	{
		int[] shift = {0,0};
		
		double[] corr = {0,0,0};
		for (int xShift = -maxShift[0]; xShift <= maxShift[0]; xShift+=maxShift[0]/10)
		{
			for (int yShift = -maxShift[1]; yShift <= maxShift[1]; yShift += maxShift[1]/10)
			{
				shift[0] = xShift;
				shift[1] = yShift;
				double tempCorr = correlation(referenceIP, targetIP,shift);
				if (tempCorr > corr[0])
				{
					corr[0] = tempCorr;
					corr[1] = shift[0];
					corr[2] = shift[1];
				}
			}	
		}
		return corr;
	}
	
	public static void main(String[] args) {
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);	
		Particle Pcenter = new Particle();
		Particle Pcenter2 = new Particle();
		int width = Math.round(12800/5);
		int height = Math.round(12800/5);
		
		short[][] referenceIPShort  = new short[width][height];					
		ByteProcessor referenceIP = new ByteProcessor(width,height);
		Pcenter.x = (int)(width/2);
		Pcenter.y = (int)(height/2);
		Pcenter2.x = (int)(width/2) + 100;
		Pcenter2.y = (int)(height/2) + 100;
		double[] drift = {0.1,0.2};
		int nFrames = 1000;
		ArrayList<Particle> referenceList = new ArrayList<Particle>();
		ArrayList<Particle> pList = new ArrayList<Particle>();
		for (int i = 0; i < nFrames; i++)
		{
			referenceIPShort[(int)(Pcenter.x+i*drift[0])][(int)(Pcenter.y+i*drift[1])] = 1;
			referenceIP.set((int)(Pcenter.x+i*drift[0]), 
					(int)(Pcenter.y+i*drift[1]), 
					referenceIP.get((int)(Pcenter.x+i*drift[0]), 
					(int)(Pcenter.y+i*drift[1])) + 10 );
			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0];
			P.y = Pcenter.y+i*drift[1];
			P.include = 1;
			P.channel = 1;
			pList.add(P);
			referenceList.add(P);
			referenceIPShort[(int)(Pcenter2.x+i*drift[0])][(int)(Pcenter2.y+i*drift[1])] = 1;
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0];
			P2.y = Pcenter2.y+i*drift[1];
			P2.include = 1;
			P2.channel = 1;
			pList.add(P2);
			referenceList.add(P2);
			
			
		}
		short[][] targetIPShort  = new short[width][height];
		ByteProcessor targetIP = new ByteProcessor(width,height);
		ArrayList<Particle> targetList = new ArrayList<Particle>();
		for (int i = nFrames; i < 2*nFrames; i++)
		{
			targetIPShort[(int)(Pcenter.x+i*drift[0])][(int)(Pcenter.y+i*drift[1])] = 1;
			targetIP.set((int)(Pcenter.x+i*drift[0]), 
					(int)(Pcenter.y+i*drift[1]), 
					targetIP.get((int)(Pcenter.x+i*drift[0]), 
					(int)(Pcenter.y+i*drift[1])) + 10);
			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0];
			P.y = Pcenter.y+i*drift[1];
			P.include = 1;
			P.channel = 2;
			pList.add(P);
			targetList.add(P);
			targetIPShort[(int)(Pcenter2.x+i*drift[0])][(int)(Pcenter2.y+i*drift[1])] = 1;
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0];
			P2.y = Pcenter2.y+i*drift[1];
			P2.include = 1;
			P2.channel = 2;
			pList.add(P2);
			targetList.add(P2);
		}
		boolean[] renderCh = new boolean[2];
		renderCh[0] = true;
		renderCh[1] = true;
		int[] pixelSize = {1,1};
		boolean gSmoothing = true;
		referenceIP.blurGaussian(2);
		targetIP.blurGaussian(2);
		generateImage.create("", renderCh, pList, width, height, pixelSize, gSmoothing); // show test data.
	
				
		
		ArrayList<Particle> pList2 = new ArrayList<Particle>();
		int[] maxShift = {250,250};
		long time1 = System.nanoTime();
		double[] shift = minimize(referenceIPShort, targetIPShort, maxShift);
		time1 = System.nanoTime() - time1;
		long time2 = System.nanoTime(); 
		double[] shift2 = minimize(referenceIP, targetIP, maxShift);
		time2 = System.nanoTime()- time2;
		long time3 = System.nanoTime(); 
		double[] shift3 = minimize(referenceList, targetList, maxShift);
		time3 = System.nanoTime()- time3;
		System.out.println(time1*1E-6 + " vs " + time2*1E-6 + " vs " + time3*1E-6);
		System.out.println(shift[1] + " " + shift[2] + " vs "+ shift3[1] + " " + shift3[2]);
		for (int i = 0; i < nFrames; i++)
		{
			
			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0];
			P.y = Pcenter.y+i*drift[1];
			P.include = 1;
			P.channel = 1;
			pList2.add(P);
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0];
			P2.y = Pcenter2.y+i*drift[1];
			P2.include = 1;
			P2.channel = 1;
			pList2.add(P2);
		}
		for (int i = nFrames; i < 2*nFrames; i++)
		{
			
			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0] + shift3[1];
			P.y = Pcenter.y+i*drift[1] + shift3[2];
			P.include = 1;
			P.channel = 2;
			pList2.add(P);
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0]+ shift3[1];
			P2.y = Pcenter2.y+i*drift[1]+ shift3[2];
			P2.include = 1;
			P2.channel = 2;
			pList2.add(P2);
		}
		
		generateImage.create("", renderCh, pList2, width, height, pixelSize, gSmoothing); // show test data.
	}

}
