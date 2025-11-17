type PassageImageProps = {
  imageUrl?: string;
  description?: string;
};

export function PassageImage({ imageUrl, description }: PassageImageProps) {
  return (
    <section className="panel passage-image">
      <header>
        <h3>Visual Illustration</h3>
      </header>
      {imageUrl ? (
        <figure>
          <img src={imageUrl} alt={description || "Generated illustration"} />
          {description && <figcaption>{description}</figcaption>}
        </figure>
      ) : (
        <p>An illustration of the passage will be displayed here.</p>
      )}
    </section>
  );
}
