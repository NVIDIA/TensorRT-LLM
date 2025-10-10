cd /usr/local/lib/python3.12/dist-packages/
ls -la | grep pytorch_triton
mv pytorch_triton-3.3.1+gitc8757738.dist-info triton-3.3.1+gitc8757738.dist-info
cd triton-3.3.1+gitc8757738.dist-info
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la
sed -i 's/^Name: pytorch-triton/Name: triton/' METADATA
sed -i 's|pytorch_triton-3.3.1+gitc8757738.dist-info/|triton-3.3.1+gitc8757738.dist-info/|g' RECORD
echo "METADATA after update:"
grep "^Name:" METADATA;
