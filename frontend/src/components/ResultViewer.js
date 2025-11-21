import React, { useRef, useEffect } from "react";

export default function ResultViewer({ result, file }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!result || !file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      result.detections.forEach((det) => {
        const { x1, y1, x2, y2 } = det.bbox;
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillText(
          `${det.class} (${det.confidence})`,
          x1,
          y1 > 20 ? y1 - 5 : y1 + 20
        );
      });
    };
  }, [result, file]);

  if (!result) return null;

  return (
    <div style={{ marginTop: "20px", textAlign: "center" }}>
      <canvas ref={canvasRef} style={{ maxWidth: "100%" }} />
    </div>
  );
}
