package de.hanslovsky.regionmerging;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.serializer.KryoRegistrator;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Test;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import de.hanslovsky.graph.edge.Edge;
import de.hanslovsky.graph.edge.EdgeCreator;
import de.hanslovsky.graph.edge.EdgeCreator.NoDataSerializableCreator;
import de.hanslovsky.graph.edge.EdgeMerger;
import de.hanslovsky.graph.edge.EdgeMerger.MAX_AFFINITY_MERGER;
import de.hanslovsky.regionmerging.BlockedRegionMergingSpark.Data;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.custom_hash.TObjectLongCustomHashMap;
import gnu.trove.map.hash.TCustomHashMap;
import gnu.trove.map.hash.TLongLongHashMap;
import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.set.hash.TLongHashSet;
import gnu.trove.strategy.HashingStrategy;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.SingleCellArrayImg;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.util.Pair;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import scala.Tuple2;

public class DataPreparationTest
{
	private static final SparkConf conf = new SparkConf()
			.setMaster( "local[1]" )
			.setAppName( DataPreparation.class.toString() )
			.set( "spark.serializer", "org.apache.spark.serializer.KryoSerializer" )
			.set( "spark.kryo.registrator", Registrator.class.getName() );

	private static final JavaSparkContext sc = new JavaSparkContext( conf );

	@AfterClass
	public static void closeSparkContext()
	{
		sc.close();
	}

	@Test
	public void testEdgeAndNodeCounts()
	{
		final long[] dimensions = { 10, 8 };
		final int[] blockSize = { 4, 5 };
		final int nDim = blockSize.length;

		final CellLoader< LongType > labelLoader = new LabelLoader( dimensions );

		final CellLoader< DoubleType > affinitiesLoader = new AffinitiesLoader( dimensions );

		final TestLoader loader = new TestLoader( dimensions, blockSize, labelLoader, affinitiesLoader );


		final List< Tuple2< HashWrapper< long[] >, Data > > graph = DataPreparation.createGraph( sc, loader, new EdgeCreator.NoDataSerializableCreator(), new EdgeMerger.MAX_AFFINITY_MERGER(), blockSize ).collect();


		final long[][] blockMinima = DataPreparation.collectAllOffsets( dimensions, blockSize, Function.identity() ).stream().toArray( long[][]::new );
		// streamception
		final long[][] blockIndices = Arrays.stream( blockMinima ).map( m -> IntStream.range( 0, nDim ).mapToLong( d -> m[ d ] / blockSize[ d ] ).toArray() ).toArray( long[][]::new );

		final HashWrapper< long[] >[] keys = Arrays.stream( blockIndices ).map( HashWrapper::longArray ).toArray( HashWrapper[]::new );

		final long[][] blockSizes = Arrays.stream( blockMinima ).map(
				m -> IntStream.range( 0, nDim )
				.mapToLong( d -> Math.min( m[ d ] + blockSize[ d ], dimensions[ d ] ) - m[ d ] )
				.toArray() )
				.toArray( long[][]::new );

		final long[] nodeCounts = Arrays.stream( blockSizes ).mapToLong( bs -> Arrays.stream( bs ).reduce( 1, ( l1, l2 ) -> l1 * l2 ) ).toArray();

		final HashingStrategy< HashWrapper< long[] > > hs = new HashingStrategy< HashWrapper< long[] > >()
		{

			@Override
			public int computeHashCode( final HashWrapper< long[] > object )
			{
				return object.hashCode();
			}

			@Override
			public boolean equals( final HashWrapper< long[] > o1, final HashWrapper< long[] > o2 )
			{
				return o1.equals( o2 );
			}
		};

		final TCustomHashMap< HashWrapper< long[] >, long[] > blockSizesMap = new TCustomHashMap<>( hs );
		final TObjectLongCustomHashMap< HashWrapper< long[] > > nodeCountsMap = new TObjectLongCustomHashMap<>( hs );
		IntStream.range( 0, keys.length ).forEach( i -> blockSizesMap.put( keys[ i ], blockSizes[ i ] ) );
		IntStream.range( 0, keys.length ).forEach( i -> nodeCountsMap.put( keys[ i ], nodeCounts[ i ] ) );

		final TObjectLongCustomHashMap< HashWrapper< long[] > > totalEdgeCountMap = new TObjectLongCustomHashMap<>( hs );
		final TObjectLongCustomHashMap< HashWrapper< long[] > > nonContractingEdgeCountMap = new TObjectLongCustomHashMap<>( hs );
		nodeCountsMap.forEachEntry( (key, value) -> {
			final long[] bs = blockSizesMap.get( key );
			final boolean isSandwiched = key.getData()[ 0 ] == 1;
			final long additionalEdges = isSandwiched ? bs[ 1 ] : 0;
			final long totalEdgeCount = value * 2 + additionalEdges;
			final long nonContractingEdgeCount = Arrays.stream( bs ).sum() + additionalEdges;
			totalEdgeCountMap.put( key, totalEdgeCount );
			nonContractingEdgeCountMap.put( key, nonContractingEdgeCount );
			return true;
		});

// check that all blocks are there
		Assert.assertEquals( blockIndices.length, graph.size() );
		Assert.assertEquals(
				Arrays.stream( keys ).collect( Collectors.toCollection( HashSet::new ) ),
				graph.stream().map( Tuple2::_1 ).collect( Collectors.toCollection( HashSet::new ) ) );

		for ( final Tuple2< HashWrapper< long[] >, Data > g : graph )
		{
			final long[] min = IntStream.range( 0, dimensions.length ).mapToLong( d -> g._1().getData()[ d ] * blockSize[ d ] ).toArray();
			final long[] max = IntStream.range( 0, dimensions.length ).mapToLong( d -> Math.min( min[ d ] + blockSize[ d ], dimensions[ d ] ) - 1 ).toArray();

//			System.out.println( Arrays.toString( min ) + " " + Arrays.toString( max ) );
			final FinalInterval interval = new FinalInterval( min, max );
			final long nEdges = 2 * Intervals.numElements( interval ) - interval.dimension( 0 ) - interval.dimension( 1 );
			final Edge e = new Edge( g._2().edges(), 0 );

//			System.out.println( "ABSURD!" + " " + g._2().nonContractingEdges() );
			final TIntHashSet nonContractingEdges = new TIntHashSet();
			g._2().nonContractingEdges().values().forEach( nonContractingEdges::addAll );
			int contractingEdgeCount = 0;
			final int edgeCount = e.size();
			int nonContractingEdgeCount = 0;

			final TLongHashSet innerNodes = new TLongHashSet();

			for ( int i = 0; i < edgeCount; ++i )
			{
				e.setIndex( i );
				Assert.assertEquals( 1, e.multiplicity() );
				Assert.assertEquals( Math.min( e.from(), e.to() ), e.affinity(), 0.0 );
				Assert.assertTrue( e.isValid() );
				Assert.assertTrue( e.isStale() );
				if ( nonContractingEdges.contains( i ) )
					++nonContractingEdgeCount;
				else
				{
					++contractingEdgeCount;
					innerNodes.add( e.from() );
					innerNodes.add( e.to() );
				}
			}

			// number of nodes
			Assert.assertEquals( nodeCountsMap.get( g._1() ), innerNodes.size() );
			// number of edges in each subgraph
			Assert.assertEquals( totalEdgeCountMap.get( g._1() ), edgeCount );
			// number of edges across subgraph broders
			Assert.assertEquals( nonContractingEdgeCountMap.get( g._1() ), nonContractingEdgeCount );
			// number of edges completely contained for each subgraph
			Assert.assertEquals( nEdges, contractingEdgeCount );
			Assert.assertEquals( edgeCount, contractingEdgeCount + nonContractingEdgeCount );
		}

	}

	@Test
	public void testVerbose()
	{
		final long[] dimensions = { 4, 3 };
		final long[] blockSize = { 2, 3 };
		final long blockCount = Intervals.numElements( blockSize );

		final ArrayImg< LongType, LongArray > labels1 = ArrayImgs.longs( LongStream.generate( () -> 1 ).limit( blockCount ).toArray(), blockSize );
		final ArrayImg< LongType, LongArray > labels2 = ArrayImgs.longs( new long[] { 2, 2, 2, 2, 3, 3 }, blockSize );

		final ArrayImg< DoubleType, DoubleArray > affinities1 = ArrayImgs.doubles( DoubleStream.generate( () -> 1.0 ).limit( blockCount * 2 ).toArray(), blockSize[ 0 ], blockSize[ 1 ], 2 );
		final ArrayImg< DoubleType, DoubleArray > affinities2 = ArrayImgs.doubles( DoubleStream.generate( () -> 1.0 ).limit( blockCount * 2 ).toArray(), blockSize[ 0 ], blockSize[ 1 ], 2 );

		final long[][] blockPositions = { { 0, 0 }, { 1, 0 } };
		final long[][] blockMinima = { { 0, 0 }, { 2, 0 } };
		final HashWrapper< long[] >[] keys = Arrays.stream( blockPositions ).map( HashWrapper::longArray ).toArray( HashWrapper[]::new );
		final HashWrapper< long[] >[] minimaKeys = Arrays.stream( blockMinima ).map( HashWrapper::longArray ).toArray( HashWrapper[]::new );

		final HashMap< HashWrapper< long[] >, ArrayImg< LongType, ? > > labelsMap = new HashMap<>();
		final HashMap< HashWrapper< long[] >, ArrayImg< DoubleType, ? > > affinitiesMap = new HashMap<>();

		labelsMap.put( minimaKeys[ 0 ], labels1 );
		labelsMap.put( minimaKeys[ 1 ], labels2 );

		affinitiesMap.put( minimaKeys[ 0 ], affinities1 );
		affinitiesMap.put( minimaKeys[ 1 ], affinities2 );

		final CellLoader< LongType > ll = cell -> burnIn( Views.translate( labelsMap.get( HashWrapper.longArray( Intervals.minAsLongArray( cell ) ) ), Intervals.minAsLongArray( cell ) ), cell );
		final CellLoader< DoubleType > al = cell -> burnIn( Views.translate( affinitiesMap.get( HashWrapper.longArray( Intervals.minAsLongArray( Views.hyperSlice( cell, 2, 0l ) ) ) ), Intervals.minAsLongArray( cell ) ), cell );
		final TestLoader loader = new TestLoader( dimensions, Arrays.stream( blockSize ).mapToInt( i -> ( int ) i ).toArray(), ll, al );

		final NoDataSerializableCreator creator = new EdgeCreator.NoDataSerializableCreator();
		final MAX_AFFINITY_MERGER merger = new EdgeMerger.MAX_AFFINITY_MERGER();

		final Map< HashWrapper< long[] >, Data > graph = DataPreparation.createGraph( sc, loader, creator, merger, Arrays.stream( blockSize ).mapToInt( l -> ( int ) l ).toArray() ).collectAsMap();

		final Data leftSubgraph = graph.get( keys[ 0 ] );
		final Data rightSubgraph = graph.get( keys[ 1 ] );

		Assert.assertEquals( 1, leftSubgraph.nonContractingEdges().size() );
		Assert.assertEquals( 1, rightSubgraph.nonContractingEdges().size() );

		Assert.assertTrue( leftSubgraph.nonContractingEdges().containsKey( keys[ 1 ] ) );
		Assert.assertTrue( rightSubgraph.nonContractingEdges().containsKey( keys[ 0 ] ) );

		Assert.assertEquals( 2, leftSubgraph.nonContractingEdges().get( keys[ 1 ] ).size() );
		Assert.assertEquals( 2, rightSubgraph.nonContractingEdges().get( keys[ 0 ] ).size() );

		final TIntHashSet nonContractingEdgesL = new TIntHashSet();
		final TIntHashSet nonContractingEdgesR = new TIntHashSet();

		leftSubgraph.nonContractingEdges().values().forEach( val -> nonContractingEdgesL.addAll( val ) );
		rightSubgraph.nonContractingEdges().values().forEach( val -> nonContractingEdgesR.addAll( val ) );

		final Edge el = new Edge( leftSubgraph.edges(), merger.dataSize() );
		final Edge er = new Edge( rightSubgraph.edges(), merger.dataSize() );

		Assert.assertEquals( 2, el.size() );
		Assert.assertEquals( 3, er.size() );

		for ( int i = 0; i < el.size(); ++i )
		{
			el.setIndex( i );
			Assert.assertTrue( "from < to: " + el, el.from() < el.to() );
			Assert.assertTrue( "edge is non-contracting", nonContractingEdgesL.contains( i ) );
			Assert.assertEquals( 1.0, el.affinity(), 0.0 );
			Assert.assertTrue( el.isValid() );
			Assert.assertTrue( el.isStale() );
			if ( el.to() == 2 )
				Assert.assertEquals( 2, el.multiplicity() );
			if ( el.to() == 3 )
				Assert.assertEquals( 1, el.multiplicity() );
		}

		for ( int i = 0; i < er.size(); ++i )
		{
			er.setIndex( i );
			Assert.assertTrue( "from < to: " + er, er.from() < er.to() );
			Assert.assertEquals( 1.0, er.affinity(), 0.0 );
			Assert.assertTrue( er.isValid() );
			Assert.assertTrue( er.isStale() );
			if ( er.from() == 1 )
			{
				Assert.assertTrue( "edge is non-contracting", nonContractingEdgesR.contains( i ) );
				if ( er.to() == 2 )
					Assert.assertEquals( 2, er.multiplicity() );
				if ( er.to() == 3 )
					Assert.assertEquals( 1, er.multiplicity() );
			}
			else if ( er.to() == 3 )
				Assert.assertEquals( 2, er.multiplicity() );
		}

	}

	private static class LabelLoader implements CellLoader< LongType >, Serializable
	{

		private final long[] dimensions;

		public LabelLoader( final long[] dimensions )
		{
			super();
			this.dimensions = dimensions;
		}

		@Override
		public void load( final SingleCellArrayImg< LongType, ? > cell ) throws Exception
		{
			final FinalInterval interval = new FinalInterval( dimensions );
			final Cursor< LongType > c = cell.localizingCursor();
			while ( c.hasNext() )
			{
				final LongType px = c.next();
				px.set( IntervalIndexer.positionToIndex( c, interval ) );
			}

		}

	}

	private static class AffinitiesLoader implements CellLoader< DoubleType >, Serializable
	{

		private final long[] dimensions;

		public AffinitiesLoader( final long[] dimensions )
		{
			super();
			this.dimensions = dimensions;
		}

		@Override
		public void load( final SingleCellArrayImg< DoubleType, ? > cell ) throws Exception
		{
			final Interval interval = new FinalInterval( dimensions );
			final int nDim = interval.numDimensions();
			for ( int d = 0; d < nDim; ++d )
			{
				final IntervalView< DoubleType > hs = Views.hyperSlice( cell, nDim, ( long ) d );
				final Cursor< DoubleType > c = hs.localizingCursor();
				while ( c.hasNext() )
				{
					final DoubleType px = c.next();
					px.set( IntervalIndexer.positionToIndex( c, interval ) );
				}
			}
		}

	}

	private static class TestLoader implements DataPreparation.Loader< LongType, DoubleType, LongArray, DoubleArray >, Serializable
	{

		private final long[] dimensions;

		private final int blockSize[];

		private final CellLoader< LongType > labelLoader;

		private final CellLoader< DoubleType > affinitiesLoader;

		public TestLoader( final long[] dimensions, final int[] blockSize, final CellLoader< LongType > labelLoader, final CellLoader< DoubleType > affinitiesLoader )
		{
			super();
			this.dimensions = dimensions;
			this.blockSize = blockSize;
			this.labelLoader = labelLoader;
			this.affinitiesLoader = affinitiesLoader;
		}

		@Override
		public CellLoader< LongType > labelLoader()
		{
			return labelLoader;
		}

		@Override
		public CellLoader< DoubleType > affinitiesLoader()
		{
			return affinitiesLoader;
		}

		@Override
		public LongType labelType()
		{
			return new LongType();
		}

		@Override
		public DoubleType affinityType()
		{
			return new DoubleType();
		}

		@Override
		public LongArray labelAccess()
		{
			return new LongArray( 0 );
		}

		@Override
		public DoubleArray affinityAccess()
		{
			return new DoubleArray( 0 );
		}

		@Override
		public CellGrid labelGrid()
		{
			return new CellGrid( dimensions, blockSize );
		}

		@Override
		public CellGrid affinitiesGrid()
		{
			final long[] d = new long[ dimensions.length + 1 ];
			final int[] b = new int[ dimensions.length + 1 ];
			System.arraycopy( dimensions, 0, d, 0, dimensions.length );
			System.arraycopy( blockSize, 0, b, 0, blockSize.length );
			d[ dimensions.length ] = dimensions.length;
			b[ dimensions.length ] = dimensions.length;
			return new CellGrid( d, b );
		}

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

	public static class TIntHashSetSerializer extends Serializer< TIntHashSet >
	{

		@Override
		public void write( final Kryo kryo, final Output output, final TIntHashSet object )
		{
			output.writeInt( object.size() );
			for ( final TIntIterator it = object.iterator(); it.hasNext(); )
				output.writeInt( it.next(), false );
		}

		@Override
		public TIntHashSet read( final Kryo kryo, final Input input, final Class< TIntHashSet > type )
		{
			final TIntHashSet set = new TIntHashSet();
			final int n = input.readInt();
			for ( int i = 0; i < n; ++i )
				set.add( input.readInt() );
			return set;
		}
	}

	public static class TDoubleArrayListSerializer extends Serializer< TDoubleArrayList >
	{

		@Override
		public void write( final Kryo kryo, final Output output, final TDoubleArrayList object )
		{
			output.writeInt( object.size() );
			output.writeDoubles( object.toArray() );
		}

		@Override
		public TDoubleArrayList read( final Kryo kryo, final Input input, final Class< TDoubleArrayList > type )
		{
			final int size = input.readInt();
			return new TDoubleArrayList( input.readDoubles( size ) );
		}
	}

	public static class TLongLongHashMapListSerializer extends Serializer< TLongLongHashMap >
	{

		@Override
		public void write( final Kryo kryo, final Output output, final TLongLongHashMap object )
		{
			final int size = object.size();
			output.writeInt( size );
			output.writeLongs( object.keys() );
			output.writeLongs( object.values() );
		}

		@Override
		public TLongLongHashMap read( final Kryo kryo, final Input input, final Class< TLongLongHashMap > type )
		{
			final int size = input.readInt();
			final long[] keys = input.readLongs( size );
			final long[] vals = input.readLongs( size );
			return new TLongLongHashMap( keys, vals );
		}
	}

	public static class HashWrapperSerializer< T > extends Serializer< HashWrapper< T > >
	{

		@Override
		public void write( final Kryo kryo, final Output output, final HashWrapper< T > object )
		{
			kryo.writeClassAndObject( output, object.getData() );
			kryo.writeClassAndObject( output, object.getHash() );
			kryo.writeClassAndObject( output, object.getEquals() );
		}

		@Override
		public HashWrapper< T > read( final Kryo kryo, final Input input, final Class< HashWrapper< T > > type )
		{
			final T data = ( T ) kryo.readClassAndObject( input );
			final ToIntFunction< T > hash = ( ToIntFunction< T > ) kryo.readClassAndObject( input );
			final BiPredicate< T, T > equals = ( BiPredicate< T, T > ) kryo.readClassAndObject( input );
			return new HashWrapper<>( data, hash, equals );
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

	public static < T extends Type< T > > void burnIn( final RandomAccessible< T > source, final RandomAccessibleInterval< T > target )
	{
		for ( final Pair< T, T > p : Views.flatIterable( Views.interval( Views.pair( source, target ), target ) ) )
			p.getB().set( p.getA() );
	}

}
