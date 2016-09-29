import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
/*
 * Contain help classes for jcuda implementation.
 */
public class CUDA {
	
	static CUdeviceptr copyToDevice(float hostData[])
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostData.length * Sizeof.FLOAT);
	    cuMemcpyHtoD(deviceData, Pointer.to(hostData), hostData.length * Sizeof.FLOAT);
	    return deviceData;
	}
	
	static CUdeviceptr copyToDevice(int hostData[])
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostData.length * Sizeof.INT);
	    cuMemcpyHtoD(deviceData, Pointer.to(hostData), hostData.length * Sizeof.INT);
	    return deviceData;
	}
	
	static CUdeviceptr allocateOnDevice(int hostDatasize)
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostDatasize * Sizeof.INT);	  
	    return deviceData;
	}
	static CUdeviceptr allocateOnDevice(short hostDatasize)
	{
	    CUdeviceptr deviceData = new CUdeviceptr();
	    cuMemAlloc(deviceData, hostDatasize * Sizeof.SHORT);	  
	    return deviceData;
	}
}
