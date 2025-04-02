#!/bin/bash

# Check if the tar.gz file already exists
if [ -f "cifar-10-python.tar.gz" ]; then
    echo "cifar-10-python.tar.gz already exists. Stopping script."
    exit 0  # Exit with success status code
fi

# If we reach here, the file doesn't exist
echo "Downloading cifar-10-python.tar.gz..."
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Now extract the file
echo "Extracting cifar-10-python.tar.gz..."
tar -xzvf cifar-10-python.tar.gz