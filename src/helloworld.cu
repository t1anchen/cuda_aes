#include <iostream>

int main(int argc, char *argv[])
{
  // Build a device property
  struct cudaDeviceProp prop;

  // Get the device property info
  cudaGetDeviceProperties(&prop, 0);

  // write to output
  std::cout << prop.name << " says: ";
  std::cout << "Hello, World!" << std::endl;

  return 0;
}
