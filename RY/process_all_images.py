import os
from DATA import preprocess_resize_save, grayscale_and_denoise, binarize_and_mask


def process_all(data_dir='static/data'):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print('No images found in', data_dir)
        return
    results = []
    for fname in files:
        print('---')
        print('Processing', fname)
        try:
            # optional: resize saved dataset copy
            try:
                preprocess_resize_save(fname)
            except Exception:
                # ignore if source already in dataset or resize fails
                pass
            try:
                grayscale_and_denoise(fname)
            except Exception as e:
                print('grayscale/denoise failed:', e)
            res, mask = binarize_and_mask(fname)
            status = 'no contours' if res is None else 'contours found'
            results.append((fname, status))
            print('Result:', status)
        except Exception as e:
            print('Error processing', fname, e)
            results.append((fname, 'error'))
    print('\nSummary:')
    for r in results:
        print(r[0], '->', r[1])


if __name__ == '__main__':
    process_all()
