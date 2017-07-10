package de.hanslovsky.regionmerging;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.IntFunction;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.serializer.KryoRegistrator;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.AccumulatorV2;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import bdv.img.h5.H5Utils;
import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import bdv.viewer.DisplayMode;
import de.hanslovsky.graph.edge.EdgeCreator;
import de.hanslovsky.graph.edge.EdgeCreator.AffinityHistogram;
import de.hanslovsky.graph.edge.EdgeMerger;
import de.hanslovsky.graph.edge.EdgeMerger.MEDIAN_AFFINITY_MERGER;
import de.hanslovsky.graph.edge.EdgeWeight;
import de.hanslovsky.graph.edge.EdgeWeight.MedianAffinityWeight;
import de.hanslovsky.regionmerging.BlockedRegionMergingSpark.Data;
import de.hanslovsky.regionmerging.BlockedRegionMergingSpark.Options;
import de.hanslovsky.regionmerging.DataPreparation.Loader;
import de.hanslovsky.util.unionfind.HashMapStoreUnionFind;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TLongArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongIntHashMap;
import gnu.trove.map.hash.TLongLongHashMap;
import gnu.trove.set.hash.TIntHashSet;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.img.cell.CellImg;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Pair;
import net.imglib2.view.Views;

public class RegionMergingExample
{

	public static String HOME_DIR = System.getProperty( "user.home" );

	public static void main( final String[] args )
	{
		final String path = HOME_DIR + "/Dropbox/misc/excerpt.h5";
		final CellImg< FloatType, ? > input = H5Utils.loadFloat( path, "main", new int[] { 300, 300, 100, 3 } );
		System.out.println( "Loaded affinities from " + path + "/main" );
		final RandomAccessibleInterval< FloatType > affs = ArrayImgs.floats( Intervals.dimensionsAsLongArray( input ) );

		final int[] perm = getFlipPermutation( input.numDimensions() - 1 );
		final RandomAccessibleInterval< FloatType > inputPerm = Views.permuteCoordinates( input, perm, input.numDimensions() - 1 );

		for ( final Pair< FloatType, FloatType > p : Views.interval( Views.pair( inputPerm, affs ), affs ) )
			p.getB().set( p.getA() );

		final RandomAccessibleInterval< LongType > labels = H5Utils.loadUnsignedLong( path, "zws", new int[] { 300, 300, 100 } );
		System.out.println( "Loaded labels from " + path + "/zws" );


		final Random rng = new Random( 100 );
		final TLongIntHashMap colors = new TLongIntHashMap();
		for ( final LongType l : Views.flatIterable( labels ) )
			if ( !colors.contains( l.get() ) )
				colors.put( l.get(), rng.nextInt() );

		final BdvStackSource< ARGBType > bdv = BdvFunctions.show( Converters.convert( Views.collapseReal( affs ), ( s, t ) -> {
			t.set( ARGBType.rgba( 255 * s.get( 0 ).get(), 255 * s.get( 1 ).get(), 255 * s.get( 2 ).get(), 1.0 ) );
		}, new ARGBType() ), "affinity", BdvOptions.options().is2D() );

		final Converter< LongType, ARGBType > colorConv = ( s, t ) -> {
			t.set( colors.get( s.get() ) );
		};
		BdvFunctions.show( Converters.convert( labels, colorConv, new ARGBType() ), "labels", BdvOptions.options().addTo( bdv ) );

		final int nBins = 256;
		final AffinityHistogram creator = new EdgeCreator.AffinityHistogram( nBins, 0.0, 1.0 );
		final MEDIAN_AFFINITY_MERGER merger = new EdgeMerger.MEDIAN_AFFINITY_MERGER( nBins );
		final MedianAffinityWeight edgeWeight = new EdgeWeight.MedianAffinityWeight( nBins, 0.0, 1.0 );

		final long[] dimensions = Intervals.dimensionsAsLongArray( labels );
		final int[] blockSize = { 300, 300, 3 };

		final CellLoader< LongType > ll = cell -> {
			burnIn( labels, cell );
		};

		final CellLoader< FloatType > al = cell -> {
			burnIn( affs, cell );
		};

		final SparkConf conf = new SparkConf()
				.setMaster( "local[*]" )
				.setAppName( DataPreparation.class.toString() )
				.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
				.set( "spark.kryo.registrator", Registrator.class.getName() );

		final JavaSparkContext sc = new JavaSparkContext( conf );

		final JavaPairRDD< HashWrapper< long[] >, Data > graph = DataPreparation.createGraph( sc, new FloatAndLongLoader( dimensions, blockSize, ll, al ), creator, merger );
		graph.cache();
		final long nBlocks = graph.count();
		System.out.println( "Starting with " + nBlocks + " blocks." );

		final MergesAccumulator accu = new MergesAccumulator();
		sc.sc().register( accu, "mergesAccumulator" );

		final IntFunction< MergeNotifyWithFinishNotification > mergeNotifyGenerator = new MergeNotifyGenerator( accu );


		final BlockedRegionMergingSpark rm = new BlockedRegionMergingSpark( merger, edgeWeight, mergeNotifyGenerator, 2 );

		final Options options = new BlockedRegionMergingSpark.Options( 0.5, StorageLevel.MEMORY_ONLY() );

		final TIntObjectHashMap< TLongArrayList > mergesLog = new TIntObjectHashMap<>();

		final Consumer< JavaPairRDD< HashWrapper< long[] >, TLongArrayList > > mergesLogger = rdd -> {
			final TLongArrayList merges = new TLongArrayList();
			rdd.values().collect().forEach( merges::addAll );
			final int newIndex = mergesLog.size();
			mergesLog.put( newIndex, merges );
		};

		System.out.println( "Start agglomerating!" );
		rm.agglomerate( sc, graph, mergesLogger, options );
		System.out.println( "Done agglomerating!" );

		final TIntObjectHashMap< TLongArrayList > merges = accu.value();

		final HashMapStoreUnionFind uf = new HashMapStoreUnionFind();
//		for ( final TLongArrayList list : merges.get( 0 ) )
		final TLongArrayList list = merges.get( 0 );
		System.out.println( "Got " + list.size() / 4 + " merges!" );
		for ( int i = 0; i < list.size(); i += 4 ) {
			final long r1 = uf.findRoot( list.get( i + 0 ) );
			final long r2 = uf.findRoot( list.get( i + 1  ) );
//			System.out.println( list.get( i ) + " " + list.get( i + 1 ) + " " + r1 + " " + r2 );
			if ( r1 != r2 )
				uf.join( r1, r2 );
		}

//		final TLongArrayList list = mergesLog.get( 0 );
//		System.out.println( "Got " + list.size() / 2 + " merges!" );
//		for ( int i = 0; i < list.size(); i += 2 )
//		{
//			final long r1 = uf.findRoot( list.get( i + 0 ) );
//			final long r2 = uf.findRoot( list.get( i + 1 ) );
////			System.out.println( list.get( i ) + " " + list.get( i + 1 ) + " " + r1 + " " + r2 );
//			if ( r1 != r2 )
//				uf.join( r1, r2 );
//		}

		final RandomAccessibleInterval< LongType > firstJoined = Converters.convert( labels, (s,t) -> { t.set( uf.findRoot( s.get() ) );}, new LongType() );
		BdvFunctions.show( Converters.convert( firstJoined, colorConv, new ARGBType() ), "after first iteration", BdvOptions.options().addTo( bdv ) );
		bdv.getBdvHandle().getViewerPanel().setDisplayMode( DisplayMode.SINGLE );
	}

	public static class FloatAndLongLoader implements Loader< LongType, FloatType, LongArray, FloatArray >
	{

		private final long[] dimensions;

		private final int[] blockSize;

		private final CellLoader< LongType > labelLoader;

		private final CellLoader< FloatType > affinitiesLoader;

		public FloatAndLongLoader( final long[] dimensions, final int[] blockSize, final CellLoader< LongType > labelLoader, final CellLoader< FloatType > affinitiesLoader )
		{
			super();
			this.dimensions = dimensions;
			this.blockSize = blockSize;
			this.labelLoader = labelLoader;
			this.affinitiesLoader = affinitiesLoader;
		}

		@Override
		public long[] dimensions()
		{
			return this.dimensions;
		}

		@Override
		public int[] blockSize()
		{
			return this.blockSize;
		}

		@Override
		public CellLoader< LongType > labelLoader()
		{
			return labelLoader;
		}

		@Override
		public CellLoader< FloatType > affinitiesLoader()
		{
			return affinitiesLoader;
		}

		@Override
		public LongType labelType()
		{
			return new LongType();
		}

		@Override
		public FloatType affinityType()
		{
			return new FloatType();
		}

		@Override
		public LongArray labelAccess()
		{
			return new LongArray( 0 );
		}

		@Override
		public FloatArray affinityAccess()
		{
			return new FloatArray( 0 );
		}

	}

	public static int[] getFlipPermutation( final int numDimensions )
	{
		final int[] perm = new int[ numDimensions ];
		for ( int d = 0, flip = numDimensions - 1; d < numDimensions; ++d, --flip )
			perm[ d ] = flip;
		return perm;
	}

	public static < T extends Type< T > > void burnIn( final RandomAccessible< T > source, final RandomAccessibleInterval< T > target )
	{
		for ( final Pair< T, T > p : Views.flatIterable( Views.interval( Views.pair( source, target ), target ) ) )
			p.getB().set( p.getA() );
	}

	public static class Registrator implements KryoRegistrator
	{

		@Override
		public void registerClasses( final Kryo kryo )
		{
//			kryo.register( HashMap.class );
//			kryo.register( TIntHashSet.class, new TIntHashSetSerializer() );
			kryo.register( Data.class, new DataSerializer() );
//			kryo.register( TDoubleArrayList.class, new TDoubleArrayListSerializer() );
//			kryo.register( TLongLongHashMap.class, new TLongLongHashMapListSerializer() );
//			kryo.register( HashWrapper.class, new HashWrapperSerializer<>() );
		}

	}

	public static class DataSerializer extends Serializer< Data >
	{

		@Override
		public void write( final Kryo kryo, final Output output, final Data object )
		{
			// edges
			output.writeInt( object.edges().size() );
			output.writeDoubles( object.edges().toArray() );

			// non-contracting edges
			output.writeInt( object.nonContractingEdges().size() );
			for ( final Entry< HashWrapper< long[] >, TIntHashSet > entry : object.nonContractingEdges().entrySet() )
			{
//				System.out.println( "Writing entry: " + entry );
				output.writeInt( entry.getKey().getData().length );
				output.writeLongs( entry.getKey().getData() );
				output.writeInt( entry.getValue().size() );
				output.writeInts( entry.getValue().toArray() );
			}

			// counts
			output.writeInt( object.counts().size() );
			output.writeLongs( object.counts().keys() );
			output.writeLongs( object.counts().values() );
		}

		@Override
		public Data read( final Kryo kryo, final Input input, final Class< Data > type )
		{
			// edges
			final int numEdges = input.readInt();
			final TDoubleArrayList edgeStore = new TDoubleArrayList( input.readDoubles( numEdges ) );

			// non-contracting edges
			final int size = input.readInt();
			final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges = new HashMap<>();
			for ( int i = 0; i < size; ++i )
			{
//				System.out.println( "reading key" );
				final int nDim = input.readInt();
				final HashWrapper< long[] > key = HashWrapper.longArray( input.readLongs( nDim ) );
//				System.out.println( "reading value" );
				final int setSize = input.readInt();
				final TIntHashSet value = new TIntHashSet( input.readInts( setSize ) );
//				System.out.println( "ok" );
				nonContractingEdges.put( key, value );
			}

			// counts
			final int numNodes = input.readInt();
			final long[] keys = input.readLongs( numNodes );
			final long[] values = input.readLongs( numNodes );
			final TLongLongHashMap counts = new TLongLongHashMap( keys, values );
			return new Data(
					edgeStore,
					nonContractingEdges,
					counts );
		}
	}

	public static class MergeNotifyGenerator implements IntFunction< MergeNotifyWithFinishNotification >, Serializable
	{

		private final MergesAccumulator merges;

		public MergeNotifyGenerator( final MergesAccumulator merges )
		{
			super();
			this.merges = merges;
		}

		@Override
		public MergeNotifyWithFinishNotification apply( final int value )
		{
			final TLongArrayList mergesInBlock = new TLongArrayList();
			return new MergeNotifyWithFinishNotification()
			{

				@Override
				public void addMerge( final long node1, final long node2, final long newNode, final double weight )
				{
					mergesInBlock.add( node1 );
					mergesInBlock.add( node2 );
					mergesInBlock.add( newNode );
					mergesInBlock.add( Double.doubleToRawLongBits( weight ) );
				}

				@Override
				public void notifyDone()
				{
					synchronized ( merges )
					{
						final TIntObjectHashMap< TLongArrayList > m = new TIntObjectHashMap<>();
						m.put( value, mergesInBlock );
						merges.add( m );
					}
//					System.out.println( "Added " + mergesInBlock.size() / 4 + " merges at iteration " + value );
				}
			};
		}

	}

	public static class MergesAccumulator extends AccumulatorV2< TIntObjectHashMap< TLongArrayList >, TIntObjectHashMap< TLongArrayList > >
	{

		private final TIntObjectHashMap< TLongArrayList > data;

		public MergesAccumulator()
		{
			this( new TIntObjectHashMap<>() );
		}

		public MergesAccumulator( final TIntObjectHashMap< TLongArrayList > data )
		{
			super();
			this.data = data;
		}

		@Override
		public void add( final TIntObjectHashMap< TLongArrayList > data )
		{
			synchronized ( this.data )
			{
				for ( final TIntObjectIterator< TLongArrayList > it = data.iterator(); it.hasNext(); )
				{
					it.advance();
					if ( !this.data.contains( it.key() ) )
						this.data.put( it.key(), new TLongArrayList() );
					this.data.get( it.key() ).addAll( it.value() );
				}
			}
		}

		@Override
		public AccumulatorV2< TIntObjectHashMap< TLongArrayList >, TIntObjectHashMap< TLongArrayList > > copy()
		{
			synchronized ( data )
			{
				final TIntObjectHashMap< TLongArrayList > copy = new TIntObjectHashMap<>( data );
				return new MergesAccumulator( copy );
			}
		}

		@Override
		public boolean isZero()
		{
			synchronized ( data )
			{
				return data.size() == 0;
			}
		}

		@Override
		public void merge( final AccumulatorV2< TIntObjectHashMap< TLongArrayList >, TIntObjectHashMap< TLongArrayList > > other )
		{
			synchronized ( data )
			{
				add( other.value() );
			}
		}

		@Override
		public void reset()
		{
			synchronized ( data )
			{
				this.data.clear();
			}
		}

		@Override
		public TIntObjectHashMap< TLongArrayList > value()
		{
			synchronized ( data )
			{
				return data;
			}
		}

	}

}
