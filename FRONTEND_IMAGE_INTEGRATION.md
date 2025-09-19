# Frontend Image Integration Guide

## Overview
The LLM is now instructed to directly generate image references using special markers that the frontend can detect and replace with actual images. No backend post-processing is needed.

## Image Marker Format
The LLM generates special markers directly in its response:

**Format**: `[IMAGE:image X.jpg]` where X is the image number

## Example Response Processing

### LLM Response (what frontend receives directly):
```
To start the machine, follow these steps:

1. Check the power switch as shown in image 1
[IMAGE:image 1.jpg]
2. Verify the pressure gauge refer to image 2
[IMAGE:image 2.jpg]
3. Press the start button see image 3
[IMAGE:image 3.jpg]
```

### No Backend Processing Needed
The LLM generates both the natural language references and the image markers directly, so the frontend receives the response ready for image embedding.

## Frontend Implementation

### 1. Parse Image Markers
```javascript
function parseImageMarkers(responseText, images) {
    // Extract all image markers from text
    const imageMarkers = responseText.match(/\[IMAGE:([^\]]+)\]/g) || [];
    
    // Create mapping from filename to image data
    const imageMap = {};
    images.forEach(img => {
        imageMap[img.filename] = img;
    });
    
    return { imageMarkers, imageMap };
}
```

### 2. Replace Markers with Images
```javascript
function embedImagesInText(responseText, images) {
    const { imageMarkers, imageMap } = parseImageMarkers(responseText, images);
    
    let processedText = responseText;
    
    imageMarkers.forEach(marker => {
        // Extract filename from marker: [IMAGE:image 1.jpg] -> image 1.jpg
        const filename = marker.match(/\[IMAGE:([^\]]+)\]/)[1];
        
        if (imageMap[filename]) {
            const imageData = imageMap[filename];
            
            // Replace marker with img tag
            const imgTag = `<img src="data:${imageData.mime_type};base64,${imageData.data}" 
                                 alt="${filename}" 
                                 class="inline-image" 
                                 style="max-width: 300px; margin: 10px 0;" />`;
            
            processedText = processedText.replace(marker, imgTag);
        }
    });
    
    return processedText;
}
```

### 3. React Component Example
```jsx
function ResponseWithImages({ response, images }) {
    const processedHtml = embedImagesInText(response.response, response.images);
    
    return (
        <div 
            className="response-content"
            dangerouslySetInnerHTML={{ __html: processedHtml }}
        />
    );
}
```

## Benefits

1. **Inline Images**: Images appear exactly where they're referenced in the text
2. **Clean Integration**: No separate image section needed
3. **Contextual**: Images are embedded in their relevant context
4. **Flexible**: Frontend can style and position images as needed
5. **Accessible**: Proper alt text and semantic HTML

## Image Data Structure
Each image in the response contains:
```json
{
    "filename": "image 1.jpg",
    "data": "base64-encoded-image-data",
    "mime_type": "image/jpeg",
    "size": 12345
}
```

## LLM Response Format
The LLM generates responses with natural language references followed by image markers:

**Example patterns:**
```
Turn the main switch as shown in image 1
[IMAGE:image 1.jpg]

Check the control panel refer to image 2  
[IMAGE:image 2.jpg]

Position the dough see image 3
[IMAGE:image 3.jpg]
```

The LLM maintains readable text while providing clear image markers for the frontend to process.
