## ffmpeg pictures into mp4
```bash
ffmpeg -framerate 10 -i lattice_%03d.png -c:v libx264 -pix_fmt yuv420p -r 30 lattice_animation.mp4

```
