package de.hanslovsky.regionmerging;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import de.hanslovsky.graph.edge.Edge;
import de.hanslovsky.graph.edge.EdgeCreator;
import de.hanslovsky.graph.edge.EdgeMerger;
import de.hanslovsky.regionmerging.BlockedRegionMergingSpark.Data;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.hash.TLongIntHashMap;
import gnu.trove.map.hash.TLongLongHashMap;
import gnu.trove.map.hash.TLongObjectHashMap;
import gnu.trove.set.hash.TIntHashSet;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.Cache;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.LoadedCellCacheLoader;
import net.imglib2.cache.ref.SoftRefLoaderCache;
import net.imglib2.img.basictypeaccess.array.ArrayDataAccess;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import scala.Tuple2;

public class DataPreparation
{

	public static interface Loader< LabelType extends NativeType< LabelType >, AffinityType extends NativeType< AffinityType >, LabelAccess, AffinityAccess >
	{

		public long[] dimensions();

		public int[] blockSize();

		public CellLoader< LabelType > labelLoader();

		public CellLoader< AffinityType > affinitiesLoader();

		public LabelType labelType();

		public AffinityType affinityType();

		public LabelAccess labelAccess();

		public AffinityAccess affinityAccess();

	}

	public static < I extends IntegerType< I > & NativeType< I >, R extends RealType< R > & NativeType< R >, LA extends ArrayDataAccess< LA >, AA extends ArrayDataAccess< AA > >
	JavaPairRDD< HashWrapper< long[] >, Data > createGraph(
			final JavaSparkContext sc,
			final Loader< I, R, LA, AA > loader,
			final EdgeCreator creator,
			final EdgeMerger merger )
	{
		final long[] dimensions = loader.dimensions();
		final int nDim = dimensions.length;
		final int[] blockSize = loader.blockSize();

		final List< HashWrapper< long[] > > blocks = collectAllOffsets( dimensions, blockSize, offset -> HashWrapper.longArray( IntStream.range( 0, nDim ).mapToLong( d -> offset[ d ] / blockSize[ d ] ).toArray() ) );

		final JavaRDD< HashWrapper< long[] > > blocksRdd = sc.parallelize( blocks );

		final Broadcast< Loader< I, R, LA, AA > > loaderBC = sc.broadcast( loader );

		final JavaPairRDD< HashWrapper< long[] >, Tuple2< RandomAccessibleInterval< I >, RandomAccessibleInterval< R > > > data = blocksRdd.mapToPair( blockPosition -> createRAIs( blockPosition, loaderBC ) );

//		System.out.println( "after creating RAIS" + data.count() );

		final JavaPairRDD< HashWrapper< long[] >, Data > graphs = data.mapToPair( positionAndData -> {
			final Loader< I, R, LA, AA > ldr = loaderBC.getValue();
			final long[] dim = ldr.dimensions();
			final long[] position = positionAndData._1().getData();
			final int[] bs = ldr.blockSize();

			final long[] min = IntStream.range( 0, nDim ).mapToLong( d -> Math.max( position[ d ] * bs[ d ], 0 ) ).toArray();
			final long[] max = IntStream.range( 0, nDim ).mapToLong( d -> Math.min( min[ d ] + bs[ d ], dim[ d ] ) - 1 ).toArray();
			final long[] innerMax = IntStream.range( 0, nDim ).mapToLong( d -> Math.min( min[ d ] + bs[ d ] - 1, dim[ d ] - 1 ) - 1 ).toArray();

			final RandomAccessibleInterval< I > labels = positionAndData._2()._1();
			final RandomAccessibleInterval< R > affinities = positionAndData._2()._2();

			final TDoubleArrayList edgeStore = new TDoubleArrayList();
			final Edge e = new Edge( edgeStore, creator.dataSize() );
			final Edge dummy = new Edge( new TDoubleArrayList(), creator.dataSize() );
			final TLongObjectHashMap< TLongIntHashMap > nodeEdgeMap = new TLongObjectHashMap<>();
			final HashMap< HashWrapper< long[] >, TIntHashSet > nonContractingEdges = new HashMap<>();
			final TLongLongHashMap counts = new TLongLongHashMap();

			for ( int d = 0; d < nDim; ++d )
			{
				final long[] currentMax = max.clone();
				currentMax[ d ] = innerMax[ d ];
				final IntervalView< R > hs = Views.hyperSlice( affinities, nDim, ( long ) d );
				addEdges( labels, hs, new FinalInterval( min, currentMax ), d, nodeEdgeMap, creator, merger, e, dummy );

				if ( min[ d ] > 0 )
				{
					final long[] lowerPos = position.clone();
					lowerPos[ d ] -= 1;
					final long[] mm = min.clone();
					final long[] MM = max.clone();
					mm[ d ] -= 1;
					MM[ d ] = mm[ d ];
//					System.out.println( "Adding lower: " + Arrays.toString( mm ) + " " + Arrays.toString( MM ) + " " + Arrays.toString( position ) + Arrays.toString( min ) + " " + d );
					final TIntHashSet edgeIndices = addEdgesAcrossBorder( labels, hs, new FinalInterval( mm, MM ), d, nodeEdgeMap, creator, merger, e, dummy );
					nonContractingEdges.put( HashWrapper.longArray( lowerPos ), edgeIndices );
				}

				if ( max[ d ] < dim[ d ] - 1 )
				{
					final long[] upperPos = position.clone();
					upperPos[ d ] += 1;
//					System.out.println( "WUT WUT! " + Arrays.toString( min ) + " " + Arrays.toString( max ) );
					final long[] mm = min.clone();
					final long[] MM = max.clone();
					mm[ d ] = MM[ d ];
//					System.out.println( "Adding upper: " + Arrays.toString( mm ) + " " + Arrays.toString( MM ) + " " + Arrays.toString( position ) + Arrays.toString( min ) + "  " + d );
					final TIntHashSet edgeIndices = addEdgesAcrossBorder( labels, hs, new FinalInterval( mm, MM ), d, nodeEdgeMap, creator, merger, e, dummy );
//					System.out.println( "Adding edge indices! " + edgeIndices );
					nonContractingEdges.put( HashWrapper.longArray( upperPos ), edgeIndices );
				}

			}
//			System.out.println( "Putting " + Arrays.toString( min ) + " " + Arrays.toString( max ) + " " + nonContractingEdges );
			final Data dat = new Data( edgeStore, nonContractingEdges, counts );
//			System.out.println( "Looking " + Arrays.toString( min ) + " " + Arrays.toString( max ) + " " + dat.nonContractingEdges() );

			return new Tuple2<>( positionAndData._1(), dat );
		} );

		return setValidAndStale( graphs, merger.dataSize() );

	}

	public static < I extends IntegerType< I >, R extends RealType< R > > void addEdges(
			final RandomAccessible< I > labels,
			final RandomAccessible< R > affinities,
			final Interval interval,
			final int d,
			final TLongObjectHashMap< TLongIntHashMap > nodeEdgeMap,
			final EdgeCreator creator,
			final EdgeMerger merger,
			final Edge e,
			final Edge dummy )
	{
		final IntervalView< I > innerLabels = Views.interval( labels, interval );
		final Cursor< R > affinitiesCursor = Views.interval( affinities, interval ).cursor();
		final Cursor< I > labelsCursor = innerLabels.cursor();
		final RandomAccess< I > labelsAccess = labels.randomAccess();
		while ( affinitiesCursor.hasNext() )
		{
			final double affinity = affinitiesCursor.next().getRealDouble();
			labelsCursor.fwd();
			if ( !Double.isNaN( affinity ) )
			{
				labelsAccess.setPosition( labelsCursor );
				labelsAccess.fwd( d );
				final long l1 = labelsCursor.get().getIntegerLong();
				final long l2 = labelsAccess.get().getIntegerLong();
//				System.out.println( "WUT " + affinity + " " + l1 + " " + l2 );

//				if ( Arrays.stream( Intervals.minAsLongArray( interval ) ).filter( m -> m > 0 ).toArray().length == 0 )
//					System.out.println(
//							Arrays.toString( Intervals.minAsLongArray( interval ) ) + " " +
//									Arrays.toString( Intervals.maxAsLongArray( interval ) ) + " " +
//									new Point( labelsCursor ) + " " + new Point( labelsAccess ) + " " + labelsCursor.get() + " " + labelsAccess.get() );
				if ( l1 != l2 && l1 > 0 && l2 > 0 )
					addEdge( labelsCursor.get().getIntegerLong(), labelsAccess.get().getIntegerLong(), affinity, nodeEdgeMap, creator, merger, e, dummy );
			}
		}
	}

	public static < I extends IntegerType< I >, R extends RealType< R > > TIntHashSet addEdgesAcrossBorder(
			final RandomAccessible< I > labels,
			final RandomAccessible< R > affinities,
			final Interval interval,
			final int d,
			final TLongObjectHashMap< TLongIntHashMap > nodeEdgeMap,
			final EdgeCreator creator,
			final EdgeMerger merger,
			final Edge e,
			final Edge dummy )
	{
		final TIntHashSet nonContractingEdges = new TIntHashSet();
		final IntervalView< I > innerLabels = Views.interval( labels, interval );
		final Cursor< R > affinitiesCursor = Views.interval( affinities, interval ).cursor();
		final Cursor< I > labelsCursor = innerLabels.cursor();
		final RandomAccess< I > labelsAccess = labels.randomAccess();
//		System.out.println( "presenting " + Arrays.toString( Intervals.minAsLongArray( interval ) ) );
//		for ( long y = interval.min( 1 ); y <= interval.max( 1 ); ++y )
//		{
//			labelsAccess.setPosition( y, 1 );
//			for ( long x = interval.min( 0 ); x <= interval.max( 0 ); ++x )
//			{
//				labelsAccess.setPosition( x, 0 );
//				System.out.print( labelsAccess.get() + " " );
//
//			}
//			System.out.println( "" );
//		}
//		System.out.println( "Image presented by" );
//		System.out.println( Arrays.toString( Intervals.minAsLongArray( interval ) ) + " " + Arrays.toString( Intervals.maxAsLongArray( interval ) ) + " WEEEET" );
		while ( affinitiesCursor.hasNext() )
		{
			final double affinity = affinitiesCursor.next().getRealDouble();
//			System.out.println( "WABABABABA " + affinity + new Point( affinitiesCursor ) );
			labelsCursor.fwd();
			if ( !Double.isNaN( affinity ) )
			{
				labelsAccess.setPosition( labelsCursor );
				labelsAccess.fwd( d );
				final long l1 = labelsCursor.get().getIntegerLong();
				final long l2 = labelsAccess.get().getIntegerLong();
				if ( l1 != l2 && l1 > 0 && l2 > 0 )
				{
					final int edgeIndex = addEdge( labelsCursor.get().getIntegerLong(), labelsAccess.get().getIntegerLong(), affinity, nodeEdgeMap, creator, merger, e, dummy );
//				if ( interval.min( 0 ) == 3 && interval.min( 1 ) == 4 )
//					System.out.println( "WAS DA LOS EY? " +
//							Arrays.toString( Intervals.minAsLongArray( interval ) ) + " " +
//							Arrays.toString( Intervals.maxAsLongArray( interval ) ) + " " +
//							new Point( labelsCursor ) + " " + new Point( labelsAccess ) + " " + labelsCursor.get() + " " + labelsAccess.get() + " " + edgeIndex );
					nonContractingEdges.add( edgeIndex );
				}
			}
		}
		return nonContractingEdges;
	}

	public static int addEdge( final long first, final long second, final double affinity, final TLongObjectHashMap< TLongIntHashMap > nodeEdgeMap, final EdgeCreator creator, final EdgeMerger merger, final Edge e, final Edge dummy )
	{

		final long from = Math.min( first, second );
		final long to = Math.max( first, second );

		if ( !nodeEdgeMap.contains( from ) )
			nodeEdgeMap.put( from, new TLongIntHashMap() );

		if ( !nodeEdgeMap.contains( to ) )
			nodeEdgeMap.put( to, new TLongIntHashMap() );

		final TLongIntHashMap fromMap = nodeEdgeMap.get( from );
		final TLongIntHashMap toMap = nodeEdgeMap.get( to );
		final int edgeIndex;
		if ( fromMap.contains( to ) )
		{
			edgeIndex = fromMap.get( to );
			assert toMap.contains( from ) && toMap.get( from ) == edgeIndex;
			e.setIndex( edgeIndex );
			creator.create( dummy, Double.NaN, affinity, from, to, 1 );
//			System.out.println( dummy + " " + e );
			merger.merge( dummy, e );
			dummy.remove();
		}
		else
		{
			assert !toMap.contains( from );
			edgeIndex = e.size();
			creator.create( e, Double.NaN, affinity, from, to, 1 );
			fromMap.put( to, edgeIndex );
			toMap.put( from, edgeIndex );
		}
		return edgeIndex;
	}

	public static < T > List< T > collectAllOffsets( final long[] dimensions, final int[] blockSize, final Function< long[], T > func )
	{
		final List< T > blocks = new ArrayList<>();
		final int nDim = dimensions.length;
		final long[] offset = new long[ nDim ];
		for ( int d = 0; d < nDim; ) {
			final long[] target = offset.clone();
			blocks.add( func.apply( target ) );
			for ( d = 0; d < nDim; ++d )
			{
				offset[ d ] += blockSize[ d ];
				if ( offset[ d ] < dimensions[ d ] )
					break;
				else
					offset[ d ] = 0;
			}
		}

		return blocks;
	}

	public static < I extends NativeType< I >, R extends NativeType< R >, LA extends ArrayDataAccess< LA >, AA extends ArrayDataAccess< AA > >
	Tuple2< HashWrapper< long[] >, Tuple2< RandomAccessibleInterval< I >, RandomAccessibleInterval< R > > >
	createRAIs( final HashWrapper< long[] > blockPosition, final Broadcast< Loader< I, R, LA, AA > > loaderBC )
	{
		final Loader< I, R, LA, AA > ldr = loaderBC.getValue();
		final CellLoader< I > labelLoader = loaderBC.getValue().labelLoader();
		final CellLoader< R > affinitiesLoader = loaderBC.getValue().affinitiesLoader();
		final long[] dim = loaderBC.getValue().dimensions();
		final int[] bs = loaderBC.getValue().blockSize();
		final int nDim = dim.length;

		final CellGrid labelsGrid = new CellGrid( dim, bs );
		final CellGrid affinitiesGrid = new CellGrid( LongStream.concat( LongStream.of( dim ), LongStream.of( nDim ) ).toArray(), IntStream.concat( IntStream.of( bs ), IntStream.of( nDim ) ).toArray() );

		final Cache< Long, Cell< LA > > labelCache = new SoftRefLoaderCache< Long, Cell< LA > >().withLoader( LoadedCellCacheLoader.get( labelsGrid, labelLoader, ldr.labelType() ) );
		final Cache< Long, Cell< AA > > affinitiesCache = new SoftRefLoaderCache< Long, Cell< AA > >().withLoader( LoadedCellCacheLoader.get( affinitiesGrid, affinitiesLoader, ldr.affinityType() ) );

		final RandomAccessibleInterval< I > labels = new CachedCellImg<>( labelsGrid, ldr.labelType(), labelCache, ldr.labelAccess() );
		final RandomAccessibleInterval< R > affinities = new CachedCellImg<>( affinitiesGrid, ldr.affinityType(), affinitiesCache, ldr.affinityAccess() );

		return new Tuple2<>( blockPosition, new Tuple2<>( labels, affinities ) );
	}

	public static < K > JavaPairRDD< K, Data > setValidAndStale( final JavaPairRDD< K, Data > input, final int dataSize )
	{
		return input.mapValues( data -> {
			final Edge e = new Edge( data.edges(), dataSize );
			final int numEdges = e.size();
			for ( int i = 0; i < numEdges; ++i )
			{
				e.setIndex( i );
				e.setValid();
				e.setStale();
			}
			return data;
		} );
	}

	public static void main( final String[] args )
	{
		final long[] dimensions = { 3, 3, 4 };
		final int[] blockSize = { 2, 2, 2 };
		final List< String > offsets = collectAllOffsets( dimensions, blockSize, ( Function< long[], String > ) offset -> Arrays.toString( offset ) );
		System.out.println( offsets );
		System.out.println( offsets.size() );
	}

}
