"""Artifactory Uploader"""
import os
import urllib.request
import urllib.error
import base64
import tarfile
import tempfile
from pathlib import Path


class ArtifactoryUploader:
    """Simple Artifactory uploader for both files and directories"""
    
    def __init__(self, base_url, repo_name, username, token):
        """
        Initialize uploader
        
        Args:
            base_url: Artifactory base URL (e.g., https://artifactory.nvidia.com/artifactory)
            repo_name: Repository name (e.g., sw-tensorrt-llm-qa-generic-local)
            username: Username
            token: API token
        """
        self.base_url = base_url.rstrip('/')
        self.repo_name = repo_name
        self.username = username
        self.token = token
        
        # Create Basic Auth header
        credentials = f"{username}:{token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        self.auth_header = f"Basic {encoded_credentials}"
    
    def upload_file(self, local_path, remote_path, properties=None):
        """
        Upload a single file to Artifactory
        
        Args:
            local_path: Local file path
            remote_path: Remote target path (relative to repository root)
            properties: Optional metadata properties dict (e.g., {"gpu": "GB300", "branch": "main"})
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if local file exists
        if not os.path.isfile(local_path):
            print(f"File not found: {local_path}")
            return False
        
        # Build upload URL
        url = f"{self.base_url}/{self.repo_name}/{remote_path}"
        
        # Add properties to URL
        if properties:
            props_str = ";".join([f"{k}={v}" for k, v in properties.items()])
            url = f"{url};{props_str}"
        
        print(f"Uploading: {local_path}")
        print(f"   Target: {remote_path}")
        
        # Get file size
        file_size = os.path.getsize(local_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"   Size: {file_size_mb:.2f} MB")
        
        try:
            # Read file content
            with open(local_path, 'rb') as f:
                file_data = f.read()
            
            # Create request
            request = urllib.request.Request(url, data=file_data, method='PUT')
            request.add_header('Authorization', self.auth_header)
            request.add_header('Content-Type', 'application/octet-stream')
            
            # Send request
            with urllib.request.urlopen(request) as response:
                status_code = response.getcode()
                
                if 200 <= status_code < 300:
                    print(f"   Success (HTTP {status_code})")
                    return True
                else:
                    print(f"   Failed (HTTP {status_code})")
                    return False
        
        except urllib.error.HTTPError as e:
            print(f"   HTTP Error: {e.code} - {e.reason}")
            try:
                error_body = e.read().decode('utf-8')
                print(f"   Response: {error_body}")
            except:
                pass
            return False
        except urllib.error.URLError as e:
            print(f"   URL Error: {e.reason}")
            return False
        except Exception as e:
            print(f"   Exception: {str(e)}")
            return False
    
    def upload_directory(self, local_dir, remote_base_path, properties=None, exclude_patterns=None):
        """
        Recursively upload an entire directory (preserves directory structure)
        
        Args:
            local_dir: Local directory path
            remote_base_path: Remote base path
            properties: Optional properties for all files
            exclude_patterns: Optional list of patterns to exclude (default: None, meaning upload all files)
        
        Returns:
            dict: {'success': success_count, 'failed': failed_count, 'total': total_count}
        """
        if not os.path.isdir(local_dir):
            print(f"Directory not found: {local_dir}")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        print(f"\nScanning directory: {local_dir}")
        print(f"   Target path: {remote_base_path}")
        
        # Collect all files
        files_to_upload = []
        local_path = Path(local_dir)
        
        for file_path in local_path.rglob('*'):
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Check exclude patterns (if provided)
            should_exclude = False
            if exclude_patterns:
                for pattern in exclude_patterns:
                    file_str = str(file_path)
                    # Simple pattern matching
                    if pattern in file_str:
                        should_exclude = True
                        break
                    # Support wildcard patterns (e.g., *.pyc)
                    if pattern.startswith('*') and file_str.endswith(pattern[1:]):
                        should_exclude = True
                        break
            
            if not should_exclude:
                files_to_upload.append(file_path)
        
        print(f"   Found {len(files_to_upload)} files to upload")
        
        # Upload all files
        stats = {'success': 0, 'failed': 0, 'total': len(files_to_upload)}
        
        for i, file_path in enumerate(files_to_upload, 1):
            # Calculate relative path (preserve directory structure)
            relative_path = file_path.relative_to(local_path)
            remote_path = f"{remote_base_path}/{relative_path}".replace('\\', '/')
            
            print(f"\n[{i}/{stats['total']}]")
            
            if self.upload_file(str(file_path), remote_path, properties):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        return stats
    
    def upload(self, local_path, remote_path, properties=None, exclude_patterns=None):
        """
        Smart upload - automatically detects if path is file or directory
        
        Args:
            local_path: Local file or directory path
            remote_path: Remote target path
            properties: Optional properties
            exclude_patterns: Optional exclude patterns (only used for directories)
        
        Returns:
            bool or dict: For files returns bool, for directories returns stats dict
        """
        if os.path.isfile(local_path):
            return self.upload_file(local_path, remote_path, properties)
        elif os.path.isdir(local_path):
            return self.upload_directory(local_path, remote_path, properties, exclude_patterns)
        else:
            print(f"Path not found: {local_path}")
            return False
    
    def upload_directory_as_archive(self, local_dir, remote_path, archive_name=None, 
                                    compression='gz', properties=None, exclude_patterns=None):
        """
        Pack directory into tar archive and upload (similar to shell script behavior)
        
        Args:
            local_dir: Local directory path
            remote_path: Remote target path (directory path in Artifactory)
            archive_name: Archive filename (without extension). If None, uses directory name
            compression: Compression type ('gz', 'bz2', 'xz', or '' for no compression)
            properties: Optional properties for the archive
            exclude_patterns: Optional list of patterns to exclude from archive
        
        Returns:
            bool: True if successful, False otherwise
        
        Example:
            # Upload ./output as output.tar.gz to trtllm/test/archives/
            uploader.upload_directory_as_archive(
                local_dir='./output',
                remote_path='trtllm/test/archives',
                archive_name='output',
                compression='gz',
                properties={'gpu': 'GB300'}
            )
        """
        if not os.path.isdir(local_dir):
            print(f"Directory not found: {local_dir}")
            return False
        
        # Generate archive name if not provided
        if archive_name is None:
            archive_name = os.path.basename(os.path.abspath(local_dir))
        
        # Determine archive extension and mode
        compression_modes = {
            'gz': ('tar.gz', 'w:gz'),
            'bz2': ('tar.bz2', 'w:bz2'),
            'xz': ('tar.xz', 'w:xz'),
            '': ('tar', 'w')
        }
        
        if compression not in compression_modes:
            print(f"Unsupported compression type: {compression}")
            print(f"   Supported: gz, bz2, xz, '' (no compression)")
            return False
        
        extension, tar_mode = compression_modes[compression]
        archive_filename = f"{archive_name}.{extension}"
        
        print(f"\nCreating archive: {archive_filename}")
        print(f"   Source: {local_dir}")
        
        # Create temporary archive file
        temp_dir = tempfile.gettempdir()
        temp_archive = os.path.join(temp_dir, archive_filename)
        
        try:
            # Create tar archive
            with tarfile.open(temp_archive, tar_mode) as tar:
                # Get all files to archive
                local_path = Path(local_dir)
                files_added = 0
                
                for file_path in local_path.rglob('*'):
                    # Skip directories
                    if file_path.is_dir():
                        continue
                    
                    # Check exclude patterns
                    should_exclude = False
                    if exclude_patterns:
                        for pattern in exclude_patterns:
                            file_str = str(file_path)
                            if pattern in file_str or (pattern.startswith('*') and file_str.endswith(pattern[1:])):
                                should_exclude = True
                                break
                    
                    if not should_exclude:
                        # Calculate arcname (relative path within archive)
                        arcname = file_path.relative_to(local_path)
                        tar.add(str(file_path), arcname=str(arcname))
                        files_added += 1
            
            # Get archive size
            archive_size = os.path.getsize(temp_archive)
            archive_size_mb = archive_size / (1024 * 1024)
            print(f"   Archive created: {archive_size_mb:.2f} MB ({files_added} files)")
            
            # Upload archive
            remote_file_path = f"{remote_path}/{archive_filename}"
            print(f"\nUploading archive...")
            success = self.upload_file(temp_archive, remote_file_path, properties)
            
            # Clean up temporary file
            os.remove(temp_archive)
            
            if success:
                print(f"   Archive uploaded successfully!")
                print(f"   Location: {remote_file_path}")
            
            return success
        
        except Exception as e:
            print(f"Failed to create archive: {str(e)}")
            # Clean up temporary file if exists
            if os.path.exists(temp_archive):
                os.remove(temp_archive)
            return False