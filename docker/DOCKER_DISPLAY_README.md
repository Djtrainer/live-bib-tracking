# Docker Display Options for Live Bib Tracking

This document explains the different ways to run the Live Bib Tracking application in Docker, depending on whether you need video display functionality.

## The Problem

When running GUI applications (like OpenCV with `cv2.imshow()`) inside Docker containers, you may encounter errors like:

```
qt.qpa.xcb: could not connect to display
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
This application failed to start because no Qt platform plugin could be initialized.
```

This happens because Docker containers don't have access to the host's display system by default.

## Solutions

### 1. No Display Mode (Recommended for Production)

**Script:** `build_and_run.sh`

This runs the application without any video display, which is perfect for:
- Production environments
- Server deployments
- When you only need the processing results/logs

```bash
./build_and_run.sh
```

The application will still:
- Process the video
- Track racers and extract bib numbers
- Print live leaderboards to the console
- Generate final results

### 2. Headless Mode with Virtual Display

**Script:** `build_and_run_headless.sh`

This creates a virtual display inside the container, allowing OpenCV to work normally but without showing windows:

```bash
./build_and_run_headless.sh
```

This is useful when:
- You want OpenCV to work as if there's a display
- You're debugging display-related code
- You want to save annotated frames (if that feature is enabled)

### 3. VNC Display Mode (Recommended for GUI)

**Script:** `build_and_run_with_display.sh` or `build_and_run_vnc.sh`

This runs a VNC server inside the container, creating a virtual desktop that you can view from your Mac:

```bash
./build_and_run_with_display.sh
# or for more control:
./build_and_run_vnc.sh
```

**How to connect:**
1. Run the script and wait 10-15 seconds for the container to start
2. Open Finder and press `Cmd+K` (or Go → Connect to Server)
3. Enter: `vnc://localhost:5900`
4. Click Connect - macOS will open the built-in Screen Sharing app
5. You'll see the container's desktop with your video processing window

**No additional software required!** This uses macOS's built-in VNC client.

## Which Option Should You Use?

- **Development/Testing:** Use option 1 (no display) for fastest performance
- **Debugging OpenCV issues:** Use option 2 (headless) to test display code without GUI
- **Full GUI experience:** Use option 3 (full display) when you need to see the video

## Environment Variables

The application also respects these environment variables:
- `DISPLAY`: Set automatically by the scripts
- `YOLO_AUTOINSTALL`: Set to "false" to prevent permission issues
- `EASYOCR_MODULE_PATH`: Set to use the correct cache directory

## Troubleshooting

### Still getting Qt errors?
- Make sure you're using the updated Dockerfile with the additional GUI libraries
- Try the headless mode first to isolate display issues
- For VNC mode, ensure the container has fully started before connecting

### VNC connection issues?
- Wait 10-15 seconds after starting the container before connecting
- Make sure port 5900 isn't being used by another application
- Try connecting to `vnc://127.0.0.1:5900` instead of `localhost`
- Check that the container is still running with `docker ps`

### Can't see the video window in VNC?
- The window might be minimized - look for it in the fluxbox taskbar
- Try right-clicking on the desktop to open applications menu
- The application logs will show in the terminal where you ran the script

### Performance issues?
- Use the no-display mode for best performance
- VNC adds some overhead but provides the best GUI experience
- The virtual display (headless) mode is faster than VNC but has no visual output
- Consider reducing the video resolution or FPS for better performance in VNC mode
