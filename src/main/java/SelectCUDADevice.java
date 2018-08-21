
import ij.IJ;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;


@Plugin(type = Command.class, menuPath = "Plugins>SMLocalizer>Select CUDA Device")
public class SelectCUDADevice implements Command{

    @Parameter
    int cudaDevice = 0;

    @Parameter
    boolean writeCUDADeviceInfos = false;

    @Override
    public void run() {
        // Sets CUDA Device used for SMLocalizer
        GlobalCUDAProps.CUDADeviceIndex = cudaDevice;
        if (writeCUDADeviceInfos) {
            IJ.showMessage(CUDA.getCUDADevicesInfo());
        }
    }

}