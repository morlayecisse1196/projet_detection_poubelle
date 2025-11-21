import React, { useState } from "react";
import { Box, Button, Typography } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import api from "../api";

export default function UploadArea({ onResult }) {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await api.post("/predict/image", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      onResult(res.data, file);
    } catch (err) {
      console.error(err);
      alert("Erreur lors de l'upload");
    }
  };

  return (
    <Box
      sx={{
        border: "2px dashed #1976d2",
        borderRadius: 2,
        p: 4,
        textAlign: "center",
        bgcolor: "#f9f9f9",
      }}
    >
      <Typography variant="h6" gutterBottom>
        Upload une image ou vidéo
      </Typography>
      <input
        type="file"
        accept="image/*,video/*"
        onChange={handleFileChange}
        style={{ margin: "20px 0" }}
      />
      <Button
        variant="contained"
        startIcon={<CloudUploadIcon />}
        onClick={handleUpload}
        disabled={!file}
      >
        Envoyer
      </Button>
    </Box>
  );
}
