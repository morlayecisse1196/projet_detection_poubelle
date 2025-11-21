import React, { useState } from "react";
import { Box, Button, Typography, LinearProgress, List, ListItem, ListItemText } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import api from "../api";

export default function VideoUpload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("frame_stride", 10); // traite 1 frame sur 10
    formData.append("max_frames", 50);   // limite à 50 frames

    try {
      const res = await api.post("/predict/video", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      alert("Erreur lors de l'analyse vidéo");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ mt: 4, p: 3, border: "2px dashed #1976d2", borderRadius: 2, bgcolor: "#f9f9f9" }}>
      <Typography variant="h6" gutterBottom>
        🎥 Upload une vidéo pour détection
      </Typography>
      <input type="file" accept="video/*" onChange={handleFileChange} style={{ margin: "20px 0" }} />
      <Button
        variant="contained"
        startIcon={<CloudUploadIcon />}
        onClick={handleUpload}
        disabled={!file || loading}
      >
        Envoyer
      </Button>

      {loading && <LinearProgress sx={{ mt: 2 }} />}

      {results && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1">Résumé des détections :</Typography>
          <List>
            {results.summary.map((frame, idx) => (
              <ListItem key={idx} sx={{ borderBottom: "1px solid #ddd" }}>
                <ListItemText
                  primary={`Frame ${frame.frame_index}`}
                  secondary={
                    frame.detections.length > 0
                      ? frame.detections.map(
                          (d) => `${d.class} (${Math.round(d.confidence * 100)}%)`
                        ).join(", ")
                      : "Aucune détection"
                  }
                />
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Box>
  );
}
