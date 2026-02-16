import fitz
import os
import sys

def extract_images(pdf_path: str, out_dir: str = 'static/data') -> int:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_count = 0
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            ext = base_image.get('ext', 'png')
            out_path = os.path.join(out_dir, f'pdf_page{page_index+1}_img{img_index}.{ext}')
            with open(out_path, 'wb') as f:
                f.write(image_bytes)
            img_count += 1
            print('Saved', out_path)
    return img_count


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python extract_images_from_pdf.py <pdf_path>')
        sys.exit(1)
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print('PDF not found:', pdf_path)
        sys.exit(2)
    count = extract_images(pdf_path)
    print(f'Extracted {count} images')
